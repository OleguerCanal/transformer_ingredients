# Transformer Ingredients 🌶🌽🥕🧄

Collection of **minimal** building blocks for transformer-based neural networks in PyTorch.

__Example:__ You can create a BERT-alike model as
```python
from transformer_ingredients import TransformerEncoder
from transformer_ingredients import Linear

class Bert(pl.LightningModule):
    def __init__(self,
                input_dim,
                max_seq_len,
                d_model,
                d_ff,
                num_layers,
                num_heads,
                dropout_p=0.3):
        super().__init__()
        self.transformer_encoder = TransformerEncoder(input_dim=input_dim,
                                                      d_model=d_model,
                                                      d_ff=d_ff,
                                                      num_layers=num_layers,
                                                      num_heads=num_heads,
                                                      dropout_p=dropout_p)
        self.output_layer = Linear(d_model, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_batch, input_lens):
        x, _ = self.transformer_encoder(input_batch, input_lens)
        output = self.softmax(self.output_layer(x))
        return output
```



**NOTE:** Most of the code is adapted from Soohwan Kim repo
