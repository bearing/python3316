#Get the address of the sis3316 card and convert to hex
# 03 = 71
read -p "Enter the two digit number on the 3316 card: " num
read -p "Select an IP address ID from 2-16: " IP
printf -v hex '%x' $num
echo $hex
size=${#hex}
A="$(arp -a)"
echo $A
#a=eth0 #$(echo $A | awk '{print $NF}')
#a=enp5s0 #$(echo $A | awk '{print $NF}')
a=en7 #$(echo $A | awk '{print $NF}')
#a=enx70886b823040 #$(echo $A | awk '{print $NF}')
#a=enx70886b823040#enx0050b622c918 #enx70886b823040#enx0050b622c918
#a=lo
sudo ifconfig $a 192.168.1.1 netmask 255.255.255.0
sleep 5
#sudo ifconfig docker0 172.17.0.1 netmask 255.255.255.0
#This IP/MAC address is taken from Struck documentation
if [[ $size -gt 2 ]]
then
  shex=${hex: -2}
  sudo arp -s 192.168.1.$IP 00:00:56:31:61:$shex
else
  sudo arp -s 192.168.1.$IP 00:00:56:31:60:$hex
fi
sleep 15
echo 192.168.1.$IP 00:00:56:31:60:$hex
A="$(arp -a)"
echo $A
