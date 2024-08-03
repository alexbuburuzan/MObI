import torch
import torch.nn as nn
from functools import partial
import logging
from transformers import CLIPVisionModel, CLIPTokenizer, CLIPTextModel
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from .xf import LayerNorm, Transformer

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, classes, class_encoder_version):
        super().__init__()

        tokenizer = CLIPTokenizer.from_pretrained(class_encoder_version)
        text_model = CLIPTextModel.from_pretrained(class_encoder_version)

        if torch.cuda.is_available():
            text_model.cuda()

        class_texts = ["a " + c if c != "empty" else c for c in classes]
        inputs = tokenizer(class_texts, return_tensors="pt", padding=True, truncation=True)

        inputs = {k: v.to(text_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_embeddings = text_model(**inputs).last_hidden_state

        # CLS token
        self.text_embeddings = text_embeddings[:, 0, :]

    def forward(self, c):
        c = self.text_embeddings[c.to(int)]
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(
            self,
            classes,
            conditions,
            version="openai/clip-vit-large-patch14",
    ) -> None:
        super().__init__()
        if "ref_image" in conditions:
            self.transformer = CLIPVisionModel.from_pretrained(version)
            self.final_ln = LayerNorm(1024)
            self.mapper = Transformer(1, 1024, 5, 1)
        
        if "ref_bbox" in conditions and "ref_label" in conditions:
            self.bbox_embedder = BBoxAndClassEmbedder(
                classes=classes,
                class_encoder_version=version,
            )
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        z = outputs.pooler_output
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, cond):
        ret = {}
        if "ref_image" in cond:
            ret["ref_image_token"] = self(cond["ref_image"])
        if "ref_bbox" in cond and "ref_label" in cond:
            ret["ref_bbox_token"] = self.bbox_embedder(
                cond["ref_bbox"],
                cond["ref_label"],
            )
        return ret
    
class BBoxAndClassEmbedder(AbstractEncoder):
    def __init__(
        self,
        classes,
        class_encoder_version="openai/clip-vit-large-patch14",
        embedder_num_freqs=4,
        proj_dims=[768, 512, 512, 768],
    ):
        super().__init__()
        self.fourier_embedder = get_embedder(
            input_dims=3,
            num_freqs=embedder_num_freqs,
        )
        self.class_embedder = ClassEmbedder(
            classes=classes,
            class_encoder_version=class_encoder_version,
        )
        text_embedding_dim = self.class_embedder.text_embeddings.shape[-1]

        self.bbox_proj = nn.Linear(
            self.fourier_embedder.out_dim * 8, proj_dims[0])
        self.second_linear = nn.Sequential(
            nn.Linear(proj_dims[0] + text_embedding_dim, proj_dims[1]),
            nn.SiLU(),
            nn.Linear(proj_dims[1], proj_dims[2]),
            nn.SiLU(),
            nn.Linear(proj_dims[2], proj_dims[3]),
        )
        
    def forward(self, bbox, class_label):
        bbox_embed = self.fourier_embedder(bbox).reshape(
            bbox.shape[0], -1).type_as(self.bbox_proj.weight)
        bbox_embed = self.bbox_proj(bbox_embed)

        class_embed = self.class_embedder(class_label)

        x = torch.cat([bbox_embed, class_embed], dim=-1)
        x = self.second_linear(x)
        return x.unsqueeze(1)
    
    def encode(self, cond):
        return {
            "ref_bbox_token": self(cond["ref_bbox"], cond["ref_label"]),
        }

class Embedder:
    """
    borrow from
    https://github.com/zju3dv/animatable_nerf/blob/master/lib/networks/embedder.py
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(input_dims, num_freqs, include_input=True, log_sampling=True):
    embed_kwargs = {
        "input_dims": input_dims,
        "num_freqs": num_freqs,
        "max_freq_log2": num_freqs - 1,
        "include_input": include_input,
        "log_sampling": log_sampling,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    logging.debug(f"embedder out dim = {embedder_obj.out_dim}")
    return embedder_obj



if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)