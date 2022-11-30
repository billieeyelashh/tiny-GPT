#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include "tingpt/model.cpp"
#include "tingpt/train.cpp"
#include "tingpt/data.cpp"

using namespace std;

int main(int argc, char** argv) {
    
    int block_size = 128;
    string text;
    ifstream inFile;
    inFile.open("input.txt");
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); 
    }
    stringstream strStream;
    strStream << inFile.rdbuf(); 
    text = strStream.str(); 
    CharDataset train_dataset(text, block_size);

    GPTConfig model_config(train_dataset.vocab_size, train_dataset.block_size, 8, 8, 512);
    TrainerConfig train_config(2, 256, 6e-4, true, 512*20, 2*train_dataset.size()*block_size, 4, "checkpoint.th");

    if (argc > 1 && string(argv[1]) == "test") {
        string output = test("I'm crying", model_config, train_config, train_dataset);
        cout << output << endl;
    }
    else if (argc > 1 && string(argv[1]) == "train") {
        train(model_config, train_config, train_dataset);
    }
    else {
        cout << "Please specify a valid command." << endl;
    }
 
    return 0;
}