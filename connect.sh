#Get the address of the sis3316 card and convert to hex
# 03 = 71
read -p "Enter the two digit number on the 3316 card: " num
printf -v hex '%x' $num
echo $hex
A="$(arp -a)"
echo $A
#a=eth0 #$(echo $A | awk '{print $NF}')
a=enp1s0 #$(echo $A | awk '{print $NF}')
a=enx70886b823040 #$(echo $A | awk '{print $NF}')
#a=enx70886b823040#enx0050b622c918 #enx70886b823040#enx0050b622c918
#a=lo
sudo ifconfig $a 192.168.1.1 netmask 255.255.255.0
sleep 15
#sudo ifconfig docker0 172.17.0.1 netmask 255.255.255.0
#This IP/MAC address is taken from Struck documentation
sudo arp -i $a -s 192.168.1.10 00:00:56:31:60:$hex
sleep 15
echo 192.168.1.10 00:00:56:31:60:$hex
A="$(arp -a)"
echo $A

