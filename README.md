# Awesome-Large-Speech-Model
A repository used to organize content related to Large Speech(Audio) Model, including paper, data, applications, tools and so on.

This repo will be updated continuously ...

## outline
- [Papers](#paper)
  - [Survey](#paper1)
  - [Speech/Audio-Extended LLM](#paper2)
  - [Speech Recognition & Translation](#paper3)
  - [Speech Synthesis](#paper4)
  - [Self-supervised Pretrain Speech Model](#paper5)
  - [Speech Tokenization/Discretization](#paper6)
  - [Audio/Music/Others](#paper7)
- [Open-source models](#model)
- [Data](#data)
- [Tools](#tool)
- [Applications](#application)

## Papers<a id="paper"></a>
### Survey<a id="paper1"></a>
- **A Comprehensive Review of Multimodal Large Language Models: Performance and Challenges Across Different Tasks** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.01319)]
- **MM-LLMs: Recent Advances in MultiModal Large Language Models** , 2024 , ACL [[paper](https://arxiv.org/pdf/2401.13601)] [[github.io](https://mm-llms.github.io/)]
- **A Survey on Multimodal Large Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.13549)] [[github](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)]
- **Sparks of Large Audio Models: A Survey and Outlook** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.12792)]

### Speech/Audio-Extended Large Language Model<a id="paper2"></a>
- **Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.16725)] [[github](https://github.com/gpt-omni/mini-omni)]
- **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context** , 2024 , arXiv [[paper](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf)] 
- **The Llama 3 Herd of Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.21783)]
- **Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2405.05945)] [[github](https://github.com/Alpha-VLLM/Lumina-T2X)]
- **VITA: Towards Open-Source Interactive Omni Multimodal LLM** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.05211)] [[github.io](https://vita-home.github.io/)]
- **Qwen2-Audio Technical Report** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.10759)] [[github](https://github.com/QwenLM/Qwen2-Audio)]
- **NExT-GPT: Any-to-Any Multimodal LLM** , 2024 , ICML [[paper](https://arxiv.org/pdf/2309.05519)] [[github](https://github.com/NExT-GPT/NExT-GPT)]
- **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling** , 2024 , arXiv [[paper](https://arxiv.org/abs/2402.12226)] [[github](https://github.com/OpenMOSS/AnyGPT)]
- **An Embarrassingly Simple Approach for LLM with Strong ASR Capacity** ， 2024 ， arXiv [[paper](An Embarrassingly Simple Approach for LLM with Strong ASR Capacity)] [[github](https://github.com/X-LANCE/SLAM-LLM)]
- **SALMONN: Towards Generic Hearing Abilities for Large Language Models** , 2024 , ICLR [[paper](https://openreview.net/pdf?id=14rn7HpKVk)] [[github](https://github.com/bytedance/SALMONN)]
- **LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2310.04673)] [[github.io](https://lauragpt.github.io/)] 
- **Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2311.07919)] [[github](https://github.com/QwenLM/Qwen-Audio)]
- **SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities** , 2023 , EMNLP [[paper](https://arxiv.org/pdf/2305.11000)] [[github](https://github.com/0nutation/SpeechGPT)]
- **Pengi: An Audio Language Model for Audio Tasks** , 2023 , NeurIPS [[paper](https://arxiv.org/pdf/2305.11834)] [[github](https://github.com/microsoft/Pengi)]
- **AudioPaLM: A Large Language Model That Can Speak and Listen** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.12925)] [[github.io](https://google-research.github.io/seanet/audiopalm/examples/)]

### Speech Recognition & Translation<a id="paper3"></a>
- **Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.04675)] [[github.io](https://bytedancespeech.github.io/seedasr_tech_report/)]
- **OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2401.16658)] [[github](https://github.com/espnet/espnet/tree/master/egs2/owsm_v3.1/s2t1)]
- **Seamless: Multilingual Expressive and Streaming Speech Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2312.05187)] [[github](https://github.com/facebookresearch/seamless_communication)] , Remark: SeamlessM4T v2
- **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.11596)] [[github](https://github.com/facebookresearch/seamless_communication)]
- **Robust Speech Recognition via Large-Scale Weak Supervision** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2212.04356)] [[github](https://github.com/openai/whisper)] , Remark: Whisper

### Speech Synthesis<a id="paper4"></a>
- **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.02430)] [[github.io](https://bytedancespeech.github.io/seedtts_tech_report/)]
- **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2403.03100)] [[github](https://speechresearch.github.io/naturalspeech3/)]
- **Boosting Large Language Model for Speech Synthesis: An Empirical Study** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2401.00246)]
- **SoundStorm: Efficient Parallel Audio Generation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2305.09636)] [[github.io](https://google-research.github.io/seanet/soundstorm/examples/)]
- **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.02111)] [[github](https://github.com/lifeiteng/vall-e)] , Remark: VALL-E
- **PromptTTS: Controllable Text-to-Speech with Text Descriptions** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2211.12171)] [[github.io](https://speechresearch.github.io/prompttts/)]

### Self-supervised Pretrain Speech Model<a id="paper5"></a>
- **BEATs: Audio Pre-Training with Acoustic Tokenizers** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2212.09058)] [[github](https://github.com/microsoft/unilm/tree/master/beats)]
- **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2202.03555)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)]
- **W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training** , 2021 , arXiv [[paper](https://arxiv.org/pdf/2108.06209)] 
- **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing** , 2021 , arXiv [[paper](https://arxiv.org/abs/2110.13900)]
- **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units** , 2021 , arXiv [[paper](https://arxiv.org/pdf/2106.07447)]
- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** , 2020 , NeurIPS [[paper](https://arxiv.org/pdf/2006.11477)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]
- **wav2vec: Unsupervised Pre-Training for Speech Recognition** , 2019 , arXiv [[paper](https://arxiv.org/pdf/1904.05862)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]

### Speech Tokenization/Discretization<a id="paper6"></a>
- **WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.16532)] [[github](https://github.com/jishengpeng/WavTokenizer)] [[github.io](https://wavtokenizer.github.io/)]
- **RepCodec: A Speech Representation Codec for Speech Tokenization** , 2024 , ACL [[paper](https://arxiv.org/pdf/2309.00169)] [[github](https://github.com/mct10/RepCodec)]
- **FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2309.07405)] [[github](https://github.com/modelscope/FunCodec)] [[github.io](https://funcodec.github.io/)]
- **Single-Codec: Single-Codebook Speech Codec towards High-Performance Speech Generation** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2406.07422)] [[github.io](https://kkksuper.github.io/Single-Codec/)]
- **ToneUnit: A Speech Discretization Approach for Tonal Language Speech Synthesis** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.08989)] [[github.io](https://toneunit1225.github.io/)]
- **SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models** , 2024 , ICLR [[paper](https://arxiv.org/pdf/2308.16692)] [[github](https://github.com/ZhangXInFD/SpeechTokenizer)]
- **Finite Scalar Quantization: VQ-VAE Made Simple** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2309.15505)] [[github](https://github.com/google-research/google-research/tree/master/fsq)]
- **High-Fidelity Audio Compression with Improved RVQGAN** , 2023 , NeurIPS [[paper](https://arxiv.org/pdf/2306.06546)] [[github](https://github.com/descriptinc/descript-audio-codec)]
- **High Fidelity Neural Audio Compression** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2210.13438)] [[github](https://github.com/facebookresearch/encodec)] , Remark: Encodec
- **vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations** , 2019 , arXiv [[paper](https://arxiv.org/pdf/1910.05453)]
- **Neural Discrete Representation Learning** , 2017 , NeurIPS, [[paper](https://arxiv.org/pdf/1711.00937)] [[code](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)] , Remark: VQ-VAE

### Audio/Music/Others<a id="paper7"></a>
- **M2UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2311.11255)] [[github](https://github.com/shansongliu/M2UGen)]
- **SingSong: Generating musical accompaniments from singing** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.12662)] [[example](https://storage.googleapis.com/sing-song/index.html)]
- **Simple and Controllable Music Generation** , 2023 , NeurIPS , [[paper](https://arxiv.org/pdf/2306.05284)] [[github](https://github.com/facebookresearch/audiocraft)]
- **AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models** , 2023 , NeurIPS , [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/e1b619a9e241606a23eb21767f16cf81-Paper-Conference.pdf)] [[github.io](https://audioldm.github.io/)]
- **Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2304.13731)] [[github](https://github.com/declare-lab/tango)] [[github.io](https://tango-web.github.io/)]
- **MusicLM: Generating Music From Text** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.11325)] [[github.io](https://google-research.github.io/seanet/musiclm/examples/)]
- **MuLan: A Joint Embedding of Music Audio and Natural Language** , 2022 , ISMIR [[paper](https://arxiv.org/pdf/2208.12415)] [[github](https://github.com/lucidrains/musiclm-pytorch)]
- **AudioLM: a Language Modeling Approach to Audio Generation** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2209.03143)] [[github.io](https://google-research.github.io/seanet/audiolm/examples/)]

## Open-source large speech models<a id="model"></a>
- **Whisper** [[github](https://github.com/openai/whisper)]
- **seamlessM4T** [[github](https://github.com/facebookresearch/seamless_communication)]
- **SenseVoice** [[github](https://github.com/FunAudioLLM/SenseVoice)]
- **CosyVoice** [[github](https://github.com/FunAudioLLM/CosyVoice)]
- **Qwen2-Audio** [[github](https://github.com/QwenLM/Qwen2-Audio)]
- **SALMONN** [[github](https://github.com/bytedance/SALMONN?tab=readme-ov-file)]

## Data<a id="data"></a>
Only common benchmarks and large-scale training datasets are listed here.
- Speech Recognition & Translation
- Speech Synthesis
- Audio Caption

|datasets|language|task|duration/hours|
| :---- | :---: | :---: | :---: |
| Librispeech [[download](https://www.openslr.org/12)] | English | ASR | 960 |
| Aishell-1 [[dowmload](https://www.openslr.org/33/)] | Mandarin | ASR | 170 |
| GigaSpeech [[download](https://github.com/SpeechColab/GigaSpeech)] | English | ASR | 10k |
| WenetSpeech [[download](https://github.com/wenet-e2e/WenetSpeech)] | Mandarin | ASR | 10k |
| Libri-Light [[download](https://github.com/facebookresearch/libri-light)] | English | ASR | 60K |
| Common Voice [[download](https://commonvoice.mozilla.org/zh-CN/datasets)] | Multilingual | ASR | >20k |
| MuST-C/MuST-C v2 | Multilingual | ST | 500 |
| CoVoST/CoVoST-2 [[download](https://github.com/facebookresearch/covost)] | Multilingual | ST | 2880 |
| GigaST [[download](https://github.com/bytedance/neurst/tree/master/datasets/GigaST)] | English-German/Chinese | ST | 10k |
| YODAS [[download](https://huggingface.co/datasets/espnet/yodas)] | Multilingual | ASR/Self-supervised | >500k |
| MSR-86K [[download](https://huggingface.co/datasets/Alex-Song/MSR-86K)] | Multilingual | ASR | 86k |
| LibriTTS [[download](https://www.openslr.org/60/)] | English | TTS | 585 |
| WenetSpeech4TTS [[download](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS)] | Mandarin | TTS | 10k |
| Emilia [[download](https://huggingface.co/datasets/amphion/Emilia-Dataset)] | Multilingual | TTS | 101k |
| AudioCaps [[download](https://huggingface.co/datasets/d0rj/audiocaps)] | - | AC | - |

## Tools<a id="tool"></a>
- **Silero VAD**: A pre-trained enterprise-grade Voice Activity Detector. [[github](https://github.com/snakers4/silero-vad)]
- **ESPnet**: End-to-end speech processing toolkit [[github](https://github.com/espnet/espnet)]
- **Amphion**: An Open-Source Audio, Music, and Speech Generation Toolkit [[github](https://github.com/open-mmlab/Amphion)]
- **huggingface speech-to-speech**: An effort for an open-sourced and modular GPT4-o. [[github](https://github.com/huggingface/speech-to-speech)]
- **faster-whisper**: Faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models. [[github](https://github.com/SYSTRAN/faster-whisper)]
- **whisper_streaming**: Whisper realtime streaming for long speech-to-text transcription and translation. [[github](https://github.com/ufal/whisper_streaming)]
- **SLAM-LLM**: A deep learning toolkit that allows researchers and developers to train custom multimodal large language model (MLLM), focusing on Speech, Language, Audio, Music processing. [[github](https://github.com/X-LANCE/SLAM-LLM)]
- **swift**: Supports training(PreTraining/Fine-tuning/RLHF), inference, evaluation and deployment of 300+ LLMs and 50+ MLLMs (multimodal large models). [[github](https://github.com/modelscope/ms-swift)] [[DOCUMENTATION](https://swift.readthedocs.io/zh-cn/latest/index.html)]

## Applications<a id="application"></a>
- **GPT-4o** [[link](https://openai.com/index/hello-gpt-4o/)]
- **Moshi** [[link](https://moshi.chat/)]
- **heygen** [[link](https://www.heygen.com/)]
- **Hume AI** [[link](https://www.hume.ai/)]
- **Suno** [[link](https://suno.com/)]
