import sys
import oneflow as flow
from libai.config import LazyConfig
from libai.config import get_config
from libai.tokenizer import BertTokenizer
from libai.config import LazyCall
from libai.engine.default import DefaultTrainer
from libai.utils.checkpoint import Checkpointer
from libai.data.structures import DistTensorData

from model.model import ModelForSequenceClassification
from dataset import CnerDataset
import pdb



def get_global_tensor(rawdata):
    t = flow.tensor(rawdata, dtype=flow.long).unsqueeze(0)
    dtd = DistTensorData(t)
    dtd.to_global()
    return dtd.tensor

class GeneratorForEager:
    def __init__(self, config_file, checkpoint_file, vocab_file):
        cfg = LazyConfig.load(config_file)
        cfg.model._target_='model.model.ModelForSequenceClassification'
        
        self.model = DefaultTrainer.build_model(cfg).eval()
        Checkpointer(self.model).load(checkpoint_file)
        # 少了一个分词器
        self.tokenizer = BertTokenizer(vocab_file)
    
    def infer(self, sentence):
        # Encode
        pdb.set_trace()
        sentence = " ".join([word for word in sentence])
        tokens_list = self.tokenizer.tokenize(sentence)
        encoder_ids_list = [self.tokenizer.bos_id] + self.tokenizer.convert_tokens_to_ids(tokens_list) + [self.tokenizer.eos_id]
        seq_len = len(encoder_ids_list)
        encoder_input_ids = get_global_tensor(encoder_ids_list)
        encoder_states = self.model.encode(encoder_input_ids, None)

if __name__ == "__main__":
    config_file = "/workspace/CQL_BERT/libai/output/benchmark/token/config.yaml"
    checkpoint_file = "/workspace/CQL_BERT/libai/output/benchmark/token/model_final"
    vocab_file = "/workspace/CQL_BERT/libai/projects/QQP/QQP_DATA/bert-base-chinese-vocab.txt"
    
    generator = GeneratorForEager(config_file, checkpoint_file, vocab_file)

    sentence = input("sentence：\n")
    result = generator.infer(sentence)
    print("results：\n" + result)
