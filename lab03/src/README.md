# Lab3 MaskGIT for Image Inpainting
## Environment Setup
<pre><code>
conda env create -f environment.yml
conda activate maskgit
</code></pre>

## Run the code

<pre><code>
python training_transformer.py
python inpainting.py
</code></pre>
(Make sure to edit the path for the dataset or checkpoint path etc.)

## Experiment Score
<pre><code>
 cd faster-pytorch-fid
 python fid_score_gpu.py --predicted-path /path/your_inpainting_results_folder --device cuda:0
</code></pre>

## TODO
- [x] Dataset Download 
- [x] MaskGIT STAGE1 Training enc, codebook, dec... Pretrained Weight (./models/VQGAN/checkpoints/)
- [ ] MultiHeadAttetion forward (./models/Transformer/modules/layers.py class MultiHeadAttetion)
- [ ] MaskGIT STAGE2 Training Transformer (./training_transformer.py ./models/VQGAN_Transformer.py)
- [ ] Implement functions for inpainting (./inpainting.py ./models/VQGAN_Transformer.py)
- [ ] Experiment Score

  

