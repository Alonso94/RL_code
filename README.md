# RL_code
Model based RL with ENN
The results plots in images folder, with videos of the training process.<br>
The code in the mpc.py file, environments in env folder.

V-REP simulator (CoppeliaSim) is used, and needed to install.
```shell script
wget http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_5_0_Linux.tar.gz
tar -xf V-REP_PRO_EDU_V3_5_0_Linux.tar.gz
rm V-REP_PRO_EDU_V3_5_0_Linux.tar.gz
```
(Experiments with Rozum in VREP 3.5, but KUKA in CoppeliaSim-latest version)

If it is running slow try install vrtualgl
```shell script
curl -O -J -L https://sourceforge.net/projects/virtualgl/files/2.6.3/virtualgl_2.6.3_amd64.deb/download
dpkg -i virtualgl_2.6.3_amd64.deb
apt install -f
/opt/VirtualGL/bin/vglserver_config -config +s +f -t
rm virtualgl_2.6.3_amd64.deb
```

After that run
```shell script
rmmod nvidia
# if : rmmod: ERROR: Module nvidia is in use by: nvidia_uvm nvidia_modeset
modprobe -r nvidia
```
