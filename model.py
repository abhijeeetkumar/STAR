'''
from torch import nn, einsum
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor

import torch
import torch.nn.functional as F
from torch.autograd import Variable

#from BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> b h d', h = h), qkv)

        dots = einsum('i h d, j h d -> h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('h i j, i h d -> j h d', attn, v)
        out = rearrange(out, 'b h d -> b (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        print(("""
               Initializing model:
               base model:         {}.
               input_representation:     {}.
               num_class:          {}.
               num_segments:       {}.
               """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model_iframe = getattr(torchvision.models, base_model)(pretrained=True)
            self.base_model_mv = getattr(torchvision.models, 'resnet18')(pretrained=True)
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_tsn(self, num_class):

        self.feature_dim_iframe = getattr(self.base_model_iframe, 'fc').in_features
        setattr(self.base_model_iframe, 'fc', nn.Linear(self.feature_dim_iframe, num_class))

        self.feature_dim_mv = getattr(self.base_model_mv, 'fc').in_features
        setattr(self.base_model_mv, 'fc', nn.Linear(self.feature_dim_mv, num_class))

        ''setattr(self.base_model_iframe, 'conv1',
                nn.Conv2d(3, 64,
                          kernel_size=(7, 7),
                          stride=(2, 2),
                          padding=(3, 3),
                          bias=False))
        self.data_bn_iframe = nn.BatchNorm2d(3)
        ''

        setattr(self.base_model_mv, 'conv1',
                nn.Conv2d(2, 64,
                          kernel_size=(7, 7),
                          stride=(2, 2),
                          padding=(3, 3),
                          bias=False))
        self.data_bn_mv = nn.BatchNorm2d(2)

        self.bert = Transformer(dim = self.feature_dim_iframe + self.feature_dim_mv, depth = 1,
                                heads = 8, dim_head = 64, mlp_dim = self.feature_dim_iframe + self.feature_dim_mv, dropout = 0.8)

        ''
                    BERT5(self.feature_dim_iframe + self.feature_dim_mv, self.num_segments,
                          hidden = self.feature_dim_iframe + self.feature_dim_mv,
                          n_layers = 1, attn_heads = 8)
        ''

        self.final = nn.Sequential(
            nn.Linear(self.feature_dim_iframe + self.feature_dim_mv, 1024),
            nn.BatchNorm1d(1024, momentum=0.01),
            nn.ReLU(),
            nn.Linear(1024, num_class),
            #nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input):
        batch_size, seq_length, c, h, w = input.shape
        input = input.view((-1, ) + input.size()[-3:])

        output_iframe = nn.Sequential(*list(self.base_model_iframe.children())[:-1])(input[:,0:3,...])
        #print(output_iframe.shape)
        #output_iframe = output_iframe.view(output_iframe.size(0), -1)
        output_iframe = output_iframe.view(output_iframe.size(0), -1)
        #output_iframe = output_iframe.transpose(1,2)  

        output_mv = nn.Sequential(*list(self.base_model_mv.children())[:-1])(self.data_bn_mv(input[:,3:5,...]))
        #print(output_mv.shape)
        #output_mv = output_mv.view(output_mv.size(0), -1)
        output_mv = output_mv.view(output_mv.size(0), -1)
        #output_mv = output_mv.transpose(1,2)


        out = torch.cat((output_iframe, output_mv), dim=1)
        out = out.reshape(out.size(0), -1)

        out = self.bert(out)
        #out = output[:,0,:] 

        out = self.dropout(out)
        out = self.final(out)
        return out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv' or self._representation == 'iframe_mv' ))])
'''

'''
import torch
import torchvision
from torch import nn, einsum
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import Tensor
import torch.nn.functional as F

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class STAM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        num_frames,
        num_classes,
        space_depth,
        space_heads,
        space_mlp_dim,
        time_depth,
        time_heads,
        time_mlp_dim,
        representation,
        space_dim_head = 64,
        time_dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 5 * patch_size ** 2

        self._input_size = image_size 
        self._representation = representation
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b f c (h p1) (w p2) -> b f (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_cls_token = nn.Parameter(torch.randn(1, dim))
        self.time_cls_token = nn.Parameter(torch.randn(1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.space_transformer = Transformer(dim, space_depth, space_heads, space_dim_head, space_mlp_dim, dropout)
        self.time_transformer = Transformer(dim, time_depth, time_heads, time_dim_head, time_mlp_dim, dropout)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, f, n, *_ = x.shape

        # concat space CLS tokens

        space_cls_tokens = repeat(self.space_cls_token, 'n d -> b f n d', b = b, f = f)
        x = torch.cat((space_cls_tokens, x), dim = -2)

        # positional embedding

        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        # space attention

        x = rearrange(x, 'b f ... -> (b f) ...')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b f) ... -> b f ...', b = b)  # select CLS token out of each frame

        # concat time CLS tokens

        time_cls_tokens = repeat(self.time_cls_token, 'n d -> b n d', b = b)
        x = torch.cat((time_cls_tokens, x), dim = -2)

        # time attention

        x = self.time_transformer(x)

        # final mlp

        return self.mlp_head(x[:, 0])

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224


    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv' or self._representation == 'iframe_mv' ))])
'''


from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
import torchvision

import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, base_model, latent_dim):
        super(Encoder, self).__init__()
        self.resnet = getattr(torchvision.models, base_model)(pretrained=True)
        self.latent_dim = latent_dim
        self.prepare_fe_layer()

    def set_attr(self, num_channels):
        setattr(self.resnet, 'conv1',
                nn.Conv2d(num_channels, 64,
                          kernel_size=(7, 7),
                          stride=(2, 2),
                          padding=(3, 3),
                          bias=False))
        self.prepare_fe_layer()

    def prepare_fe_layer(self):
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, self.latent_dim), nn.BatchNorm1d(self.latent_dim, momentum=0.01)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x

class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(latent_repr.size(0), 1, self.hidden_attention.in_features), requires_grad=False
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        print(("""
               Initializing model:
               base model:         {}.
               input_representation:     {}.
               num_class:          {}.
               num_segments:       {}.
               """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self.base_model_iframe = getattr(torchvision.models, base_model)(pretrained=True)
            self.base_model_mv = getattr(torchvision.models, base_model)(pretrained=True)
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_tsn(self, num_class):

        feature_dim_iframe = getattr(self.base_model_iframe, 'fc').in_features
        setattr(self.base_model_iframe, 'fc', nn.Linear(feature_dim_iframe, num_class))

        feature_dim_mv = getattr(self.base_model_mv, 'fc').in_features
        setattr(self.base_model_mv, 'fc', nn.Linear(feature_dim_mv, num_class))

        '''setattr(self.base_model_iframe, 'conv1',
                nn.Conv2d(3, 64, 
                          kernel_size=(7, 7),
                          stride=(2, 2),
                          padding=(3, 3),
                          bias=False))
        self.data_bn_iframe = nn.BatchNorm2d(3)
        '''

        setattr(self.base_model_mv, 'conv1',
                nn.Conv2d(2, 64,     
                          kernel_size=(7, 7),
                          stride=(2, 2),
                          padding=(3, 3),
                          bias=False))
        self.data_bn_mv = nn.BatchNorm2d(2)

        self.final = nn.Sequential(
            nn.Linear(feature_dim_iframe+feature_dim_mv, 1024),
            nn.BatchNorm1d(1024, momentum=0.01),
            nn.ReLU(),
            nn.Linear(1024, num_class),
            #nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout()

    def forward(self, input):
        batch_size, seq_length, c, h, w = input.shape
        input = input.view((-1, ) + input.size()[-3:])
 
        output_iframe = nn.Sequential(*list(self.base_model_iframe.children())[:-1])(input[:,0:3,...])
        output_iframe = output_iframe.view(output_iframe.size(0), -1)
        output_mv = nn.Sequential(*list(self.base_model_mv.children())[:-1])(self.data_bn_mv(input[:,3:5,...]))
        output_mv = output_mv.view(output_mv.size(0), -1)

        out = torch.cat((output_iframe, output_mv), dim=1)
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.dropout(out)
        out = self.final(out)
        #print(out.shape)
        return out

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224


    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv' or self._representation == 'iframe_mv' ))])


'''
class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation, 
                 base_model='resnet152', latent_dim=512, lstm_layers=3,
                 hidden_dim=1024, bidirectional=True, attention=True):
        super(Model, self).__init__()
        self.encoder = Encoder(base_model, latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_class),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)
        self._representation = representation
        self.num_segments = num_segments
        print(("""
    Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)
        self._prepare_tsn(num_class)

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def _prepare_tsn(self, num_class):

        if self._representation == 'mv':
            self.encoder.set_attr(2)
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)
        if self._representation == 'iframe_mv':
            self.encoder.set_attr(5)
            self.data_bn = nn.BatchNorm2d(5)

    def forward(self, input):
        batch_size, seq_length, c, h, w = input.shape
        input = input.view((-1, ) + input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        #self.lstm.flatten_parameters()

        input = self.encoder(input)
        input = input.view(batch_size, seq_length, -1)
        input = self.lstm(input)

        if self.attention:
            attention_w = F.softmax(self.attention_layer(input).squeeze(-1), dim=-1)
            input = torch.sum(attention_w.unsqueeze(-1) * input, dim=1)
        else:
            input = input[:, -1]

        return self.output_layers(input)

    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
        else:
            scales = [1, .875, .75, .66]

        print('Augmentation scales:', scales)
        return torchvision.transforms.Compose(
            [GroupMultiScaleCrop(self._input_size, scales),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv' or self._representation == 'iframe_mv' ))])
'''
