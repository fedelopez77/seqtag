
from torch import nn
from seqtag.embedding_model import get_embedding_model


class SequenceTaggingModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embedding_model = get_embedding_model(args.language_model)
        embed_size = self.embedding_model.config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(embed_size, args.num_labels)

    def forward(self, tokens, masks):
        """
        :param tokens: tensor of b, seq_len
        :param masks: tensor of b, seq_len
        :return: logits: b, seq_len, num_labels
        """
        embeddings = self.embedding_model(tokens, attention_mask=masks)[0]      # b, seq_len, emb_size
        logits = self.linear(self.dropout(embeddings))                          # b, seq_len, num_labels
        return logits
