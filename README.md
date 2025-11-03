#CSOPESY OS Emulator

###Project Members:
    Apetreor, Lee Jacob Marcus
    Leander, Wendel Walter
    David, Joanna Isabelle
    Crescini, Nick Jenson

##COMPILATION INSTRUCTIONS

To compile the emulator, use:
    g++ -std=c++20 main.cpp -pthread -O2 -o main.exe
To run the emulator, use:
    .\main.exe

###Commands:
 - initialize           : load config.txt and start scheduler (required before other commands except exit)
 - exit                 : terminate console
 - screen -s <name>     : create new process and attach screen
 - screen -r <name>     : re-attach to an existing running process
 - screen -ls           : list running and finished processes, cores used/available
 - scheduler-start      : begin automatic dummy process generation
 - scheduler-stop       : stop automatic process generation
 - report-util          : generate csopesy-log.txt CPU utilization report
 - help                 : show this help
