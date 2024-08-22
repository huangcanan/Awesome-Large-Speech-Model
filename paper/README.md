# Papers Relative to Large Speech Model

## Categorization
- [Survey](#subtitle1)
- [Speech/Audio-Text LLM](#subtitle2)
- [Speech Tokenization/Discretization](#subtitle3)
- [Speech Recognition & Translation](#subtitle4)
- [Self-supervise Pretrain Speech Model](#subtitle5)
- [Speech Synthesis](#subtitle6)
- [Audio/Music/Others](#subtitle7)


## Survey<a id="subtitle1"></a>
- **A Comprehensive Review of Multimodal Large Language Models: Performance and Challenges Across Different Tasks** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2408.01319)]
- **MM-LLMs: Recent Advances in MultiModal Large Language Models** , 2024 , ACL [[paper](https://arxiv.org/pdf/2401.13601)] [[github.io](https://mm-llms.github.io/)]
- **A Survey on Multimodal Large Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.13549)] [[github](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)]
- **Sparks of Large Audio Models: A Survey and Outlook** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.12792)]

## Speech/Audio-Text Large Language Model<a id="subtitle2"></a>
- **Qwen2-Audio Technical Report** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.10759)] [[github](https://github.com/QwenLM/Qwen2-Audio)]
- **NExT-GPT: Any-to-Any Multimodal LLM** , 2024 , ICML [[paper](https://arxiv.org/pdf/2309.05519)] [[github](https://github.com/NExT-GPT/NExT-GPT)]
- **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling** , 2024 , arXiv [[paper](https://arxiv.org/abs/2402.12226)] [[github](https://github.com/OpenMOSS/AnyGPT)]
- **SALMONN: TOWARDS GENERIC HEARING ABILITIES FOR LARGE LANGUAGE MODELS** , 2024 , ICLR , [[paper](https://openreview.net/pdf?id=14rn7HpKVk)] [[github](https://github.com/bytedance/SALMONN)]
- **Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2311.07919)] [[github](https://github.com/QwenLM/Qwen-Audio)]
- **SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities** , 2023 , EMNLP [[paper](https://arxiv.org/pdf/2305.11000)] [[github](https://github.com/0nutation/SpeechGPT)]
- **Pengi: An Audio Language Model for Audio Tasks** , 2023 , NeurIPS [[paper](https://arxiv.org/pdf/2305.11834)] [[github](https://github.com/microsoft/Pengi)]
- **AudioPaLM: A Large Language Model That Can Speak and Listen** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2306.12925)] [[github.io](https://google-research.github.io/seanet/audiopalm/examples/)]

## Speech Tokenization/Discretization<a id="subtitle3"></a>
- **Single-Codec: Single-Codebook Speech Codec towards High-Performance Speech Generation** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2406.07422)] [[github.io](https://kkksuper.github.io/Single-Codec/)]
- **ToneUnit: A Speech Discretization Approach for Tonal Language Speech Synthesis** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.08989)] [[github.io](https://toneunit1225.github.io/)]
- **SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models** , 2024 , ICLR [[paper](https://arxiv.org/pdf/2308.16692)] [[github](https://github.com/ZhangXInFD/SpeechTokenizer)]
- **Finite Scalar Quantization: VQ-VAE Made Simple** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2309.15505)] [[github](https://github.com/google-research/google-research/tree/master/fsq)]
- **High Fidelity Neural Audio Compression** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2210.13438)] [[github](https://github.com/facebookresearch/encodec)] , Remark: Encodec
- **Neural Discrete Representation Learning** , 2017 , NeurIPS, [[paper](https://arxiv.org/pdf/1711.00937)] [[code](https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py)] , Remark: VQ-VAE

## Speech Recognition & Translation<a id="subtitle4"></a>
- **Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2407.04675)] [[github.io](https://bytedancespeech.github.io/seedasr_tech_report/)]
- **OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer** , 2024 , Interspeech [[paper](https://arxiv.org/pdf/2401.16658)] [[github]
- **Seamless: Multilingual Expressive and Streaming Speech Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2312.05187)] [[github](https://github.com/facebookresearch/seamless_communication)] , Remark: SeamlessM4T v2
- **SeamlessM4T: Massively Multilingual & Multimodal Machine Translation** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2308.11596)] [[github](https://github.com/facebookresearch/seamless_communication)]
- **Robust Speech Recognition via Large-Scale Weak Supervision** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2212.04356)] [[github](https://github.com/openai/whisper)] , Remark: Whisper

## Self-supervise Pretrain Speech Model<a id="subtitle5"></a>
- **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2202.03555)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)]
- **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing** , 2021 , arXiv [[paper](https://arxiv.org/abs/2110.13900)]
- **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units** , 2021 , arXiv [[paper](https://arxiv.org/pdf/2106.07447)]
- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** , 2020 , NeurIPS [[paper](https://arxiv.org/pdf/2006.11477)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]
- **wav2vec: Unsupervised Pre-Training for Speech Recognition** , 2019 , arXiv [[paper](https://arxiv.org/pdf/1904.05862)] [[github](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)]

## Speech Synthesis<a id="subtitle6"></a>
- **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2406.02430)] [[github.io](https://bytedancespeech.github.io/seedtts_tech_report/)]
(https://github.com/espnet/espnet/tree/master/egs2/owsm_v3.1/s2t1)]
- **NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models** , 2024 , arXiv [[paper](https://arxiv.org/pdf/2403.03100)] [[github](https://speechresearch.github.io/naturalspeech3/)]
- **Boosting Large Language Model for Speech Synthesis: An Empirical Study** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2401.00246)]
- **Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers** , 2023 , arXiv [[paper](https://arxiv.org/pdf/2301.02111)] [[github](https://github.com/lifeiteng/vall-e)] , Remark: VALL-E

## Audio/Music/Others<a id="subtitle7"></a>
- **AudioLM: a Language Modeling Approach to Audio Generation** , 2022 , arXiv [[paper](https://arxiv.org/pdf/2209.03143)] [[github.io](https://google-research.github.io/seanet/audiolm/examples/)]
