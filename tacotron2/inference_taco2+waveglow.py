
import matplotlib
#%matplotlib inline
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import time

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
#from denoiser import Denoiser
from scipy.io.wavfile import read, write
from plotting_utils import *

import math
import cProfile, pstats, io

import torch.autograd.profiler as profiler

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Device : ", device)

def plot_data(data, figsize=(16, 4)):
	fig, axes = plt.subplots(1, len(data), figsize=figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect='auto', origin='bottom', interpolation='none')


with profiler.profile(record_shapes=True,use_cuda=use_cuda) as prof:
	with profiler.record_function("model_inference"):

		hparams = create_hparams()
		hparams.sampling_rate = 22050

		checkpoint_path = "/home/stuart/sagar/speech_analysis_synth/tacotron2/tacotron2_statedict.pt"
		model = load_model(hparams)
		model.load_state_dict(torch.load(checkpoint_path, map_location = device)['state_dict'])
		if use_cuda:
			_ = model.cuda().eval().half()
		else:
			_ = model.to(device).eval()

		waveglow_path = '/home/stuart/sagar/speech_analysis_synth/waveglow/checkpoints_en_us_male_62k/waveglow_212000'

		waveglow = torch.load(waveglow_path,map_location=device)['model']

		if use_cuda:
			waveglow.cuda().eval().half()
		else:
			waveglow.to(device).eval()

		for k in waveglow.convinv:
			k.float()
		# optional
		#denoiser = Denoiser(waveglow)

		# text = "Scientists at the CERN laboratory say they have discovered a new particle."
		# text = "There\'s a way to measure the acute emotional intelligence that has never gone out of style."
		text = "President Trump met with other leaders at the Group of 20 conference."
		sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
		if use_cuda:
			sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
		else:
			sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()

		start_taco2 = time.time()
		mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
		print("alignments:", alignments)
		taco2_time = time.time()-start_taco2
		mel_outputs_postnet = mel_outputs_postnet.to(device)
		start_waveglow = time.time()
		with torch.no_grad():
			audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
		waveglow_time = time.time()-start_waveglow

print("-----------Report for GPU inference-------")
write('synthesized_out.wav',22050,audio.float().cpu().detach().numpy().T)

print("Tacotron2 Inference time:", taco2_time)
print("Waveglow Inference time:", waveglow_time)
print("Total Execution time :", taco2_time + waveglow_time)
print("length of audio synthesized :{} sec".format(len(audio.float().cpu().detach()[0])/22050))

