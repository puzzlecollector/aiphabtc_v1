# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Gyeongmin Kim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"spm_model": "spm.model", "custom_vocab_file": "dict.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "spm_model": {
        "fairseq-roberta-spm-normal": "fairseq-roberta-all-model/spm.model",
    },
    "custom_vocab_file": {
        "fairseq-roberta-spm-normal": "fairseq-roberta-all-model/dict.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "fairseq-roberta-spm-normal": 512,
}


class FairSeqRobertaSentencePieceTokenizer(XLMRobertaTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
            self,
            spm_model,
            custom_vocab_file,
            bos_token="[CLS]",
            eos_token="[SEP]",
            sep_token="[SEP]",
            cls_token="[CLS]",
            unk_token="[UNK]",
            pad_token="[PAD]",
            mask_token="[MASK]",
            **kwargs
    ):
        super().__init__(
            vocab_file=spm_model,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # FairSeq dictioanry: <s>, <pad>, </s>, <unk>, token1, token2, ..., tokenN, <mask>
        self.symbols = []
        self.count = []
        self.spm_id_to_fairseq_id = {}
        self._add_symbol(self.sp_model.PieceToId(bos_token))
        self._add_symbol(self.sp_model.PieceToId(pad_token))
        self._add_symbol(self.sp_model.PieceToId(eos_token))
        self._add_symbol(self.sp_model.PieceToId(unk_token))
        self._add_from_file(custom_vocab_file)
        self._add_symbol(self.sp_model.PieceToId(mask_token))

        self.fairseq_tokens_to_ids = {}
        self.fairseq_tokens_to_ids = self._build_fairseq_tokens_to_ids()
        # self.spm_id_to_fairseq_id(bridge vocab)을 이용해서 real token -> fairseq id로 연결해주는 vocabulary
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # Collect some stats like OOV rate.
        self._num_tokens_converted = 0
        self._num_tokens_oov = 0

    @property
    def vocab_size(self):
        return len(self.symbols)

    @property
    def pad_token_id(self):
        return self.fairseq_tokens_to_ids.get(self.pad_token)

    @property
    def unk_token_id(self):
        return self.fairseq_tokens_to_ids.get(self.unk_token)

    def reset_stats(self):
        self._num_tokens_converted = 0
        self._num_tokens_oov = 0

    def get_stats(self):
        oov_rate = self._num_tokens_oov / self._num_tokens_converted
        result = {
            "total": self._num_tokens_converted,
            "oov": self._num_tokens_oov,
            "oov_rate": oov_rate
        }
        return result

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        self._num_tokens_converted += 1
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        else:
            self._num_tokens_oov += 1
            return self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        else:
            return self.unk_token

    def _add_from_file(self, f):
        """
        Source: FairSeq Dictionary class.
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self._add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = 0

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                spm_id = line
                if spm_id in self.spm_id_to_fairseq_id and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file."
                        .format(spm_id)
                    )
                self._add_symbol(spm_id, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _add_symbol(self, spm_id, n=1, overwrite=False):
        """
        Source: FairSeq Dictionary class.
        Adds a word to the dictionary
        """
        if spm_id in self.spm_id_to_fairseq_id and not overwrite:
            idx = self.spm_id_to_fairseq_id[spm_id]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.spm_id_to_fairseq_id[spm_id] = idx
            self.symbols.append(spm_id)
            self.count.append(n)
            return idx

    def _build_fairseq_tokens_to_ids(self):
        # self.spm_id_to_fairseq_id(bridge vocab)을 이용해서 real token -> fairseq id로 연결해주는 vocabulary 빌드
        fairseq_tokens_to_ids = self.fairseq_tokens_to_ids
        for spm_id, fairseq_id in self.spm_id_to_fairseq_id.items():
            if isinstance(spm_id, str) and "madeup" in spm_id:
                print("[PASS] spm_id: {} | fairseq_id: {}".format(spm_id, fairseq_id))
                continue
            token = self.sp_model.IdToPiece(int(spm_id))
            # print("token: {} | spm_id: {} | fairseq_id: {}".format(token, spm_id, fairseq_id))
            fairseq_tokens_to_ids[str(token)] = fairseq_id
        return fairseq_tokens_to_ids
