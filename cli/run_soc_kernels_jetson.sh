#  On the Nano:
#  cd ~/dev/branes/clones/graphs && git pull
# sudo nvpmodel -m <MAXN_SUPER id> 
sudo nvpmodel -m 0
sudo jetson_clocks
python3 cli/benchmark_soc_kernels.py --hardware jetson_orin_nano_8gb --power-mode MAXN_SUPER

