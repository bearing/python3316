#Array of card names, purely for aesthetical purposes can be named whatever
declare -a CardNames=("Card 14" "Card 2" "Card 4" "Card 6" "Card 7" "Card 8" "Card 10" "Card 12") # CAMIS
#declare -a CardNames=("Card 3" "Card 9" "Card 11" "Card 13") # MKS PGI

#Card ID numbers, found on the back of the 3316 cards. Last 3 digits (or 2 if third to last is 0)
declare -a CardIDs=("457" "67" "69" "111" "215" "110" "54" "68") # CAMIS
#declare -a CardIDS=("71" "109" "70" "397") # MKS PGI

#IP numbers to assign the cards when running, can range from 2-16
declare -a CardIPs=("3" "4" "5" "6" "7" "8" "9" "10") # CAMIS
#declare -a CardIPS=("2" "3" "4" "5") # MKS PGI

a=enx7cc2c64b9952
echo Performing intial setup config$'\n'
sudo ifconfig $a 192.168.0.1 netmask 255.255.255.0
sleep 5

for (( i=0; i<${#CardIDs[@]}; i++ )); do
    echo Now setting up ${CardNames[$i]} \(ID \#${CardIDs[$i]}\)
    #Get the address of the sis3316 card and convert to hex
    printf -v hex '%x' ${CardIDs[$i]}

    size=${#hex}

    #This IP/MAC address is taken from Struck documentation
    if [[ $size -gt 2 ]]
    then
      shex=${hex: -2}
      sudo arp -s 192.168.0.${CardIPs[$i]} 00:00:56:31:61:$shex
    else
      sudo arp -s 192.168.0.${CardIPs[$i]} 00:00:56:31:60:$hex
    fi
    sleep 15
    echo Assigned ${CardNames[$i]} the IP address: 192.168.0.${CardIPs[$i]}
    echo '-------------------------'
done

echo $'\n'Printing the entire arp table for confirmation: $'\n'
echo $(arp -a)
