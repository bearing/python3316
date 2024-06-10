#Get the address of the sis3316 card and convert to hex
# 03 = 71
read -p "Enter the bottom number on the 3316 card: " num
read -p "Select an IP address ID from 3-16: " IP
printf -v hex '%x' $num
# echo $hex
size=${#hex}
A="$(arp -a)"
# echo $A
#a=eth0 #$(echo $A | awk '{print $NF}')
# a=enp5s0 #$(echo $A | awk '{print $NF}')
a=enx7cc2c64b9952
# a=enxc05627b11583 #$(echo $A | awk '{print $NF}')
sudo ifconfig $a 192.168.0.1 netmask 255.255.255.0
sleep 5
#sudo ifconfig docker0 172.17.0.1 netmask 255.255.255.0
#This IP/MAC address is taken from Struck documentation
if [[ $size -gt 2 ]]
then
  shex=${hex: -2}
  sudo arp -s 192.168.0.$IP 00:00:56:31:61:$shex
else
  sudo arp -s 192.168.0.$IP 00:00:56:31:60:$hex
fi
sleep 15
echo 192.168.0.$IP 00:00:56:31:60:$hex
A="$(arp -a)"
echo $A
