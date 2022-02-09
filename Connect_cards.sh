#Array of card names, purely for aesthetical purposes can be named whatever
declare -a CardNames=("Card 1" "Card 2" "Card 10" "Card 12" "Card 6" "Card 7" "Card 4" "Card 5") # CAMIS
#declare -a CardNames=("Card 3" "Card 9" "Card 11" "Card 13") # MKS PGI

#Card ID numbers, found on the back of the 3316 cards. Last 3 digits (or 2 if third to last is 0)
declare -a CardIDs=("66" "67" "54" "68" "111" "215" "69" "216") # CAMIS
#declare -a CardIDS=("71" "109" "70" "397") # MKS PGI

#IP numbers to assign the cards when running, can range from 2-16
declare -a CardIPs=("2" "3" "4" "5" "6" "7" "8" "9") # CAMIS
#declare -a CardIPS=("2" "3" "4" "5") # MKS PGI

a=enp5s0
echo Performing intial setup config$'\n'
sudo ifconfig $a 192.168.1.1 netmask 255.255.255.0
sleep 5

for (( i=0; i<${#CardIDs[@]}; i++ )); do
    echo Now setting up ${CardNames[$i]} \(ID \#${CardIDs[$i]}\)
    #Get the address of the sis3316 card and convert to hex
    printf -v hex '%x' ${CardIDs[$i]}

    #This IP/MAC address is taken from Struck documentation
    sudo arp -i $a -s 192.168.1.${CardIPs[$i]} 00:00:56:31:60:$hex
    sleep 15
    echo Assigned ${CardNames[$i]} the IP address: 192.168.1.${CardIPs[$i]}
    echo '-------------------------'
done

echo $'\n'Printing the entire arp table for confirmation: $'\n'
echo $(arp -a)
