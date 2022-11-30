#include <iostream>
#include <math.h>
#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/utils.h>

class CharDataset : public torch::data::Dataset<CharDataset> {
	public:
		CharDataset(std::string data, int block_size) {
			std::vector<char> chars = list(set(data));
			int data_size = data.size();
			int vocab_size = chars.size();
			std::cout << "Data: " << data_size << " char, " << vocab_size << " dict" << std::endl;
			stoi = std::map<char, int>();
			itos = std::map<int, char>();
			for (int i = 0; i < vocab_size; i++) {
				stoi[chars[i]] = i;
				itos[i] = chars[i];
			}
			this->block_size = block_size;
			this->vocab_size = vocab_size;
			this->data = data;
		}

		std::pair<torch::Tensor, torch::Tensor> get(size_t idx) {
			std::string chunk = data.substr(idx, idx + block_size + 1);
			std::vector<int> dix;
			for (int i = 0; i < chunk.size(); i++) {
				dix.push_back(stoi[chunk[i]]);
			}
			torch::Tensor x = torch::tensor(dix.begin(), dix.end()-1, torch::dtype(torch::kLong));
			torch::Tensor y = torch::tensor(dix.begin()+1, dix.end(), torch::dtype(torch::kLong));
			return std::make_pair(x, y);
		}

		torch::optional<size_t> size() const {
			return data.size() - block_size;
		}

	private:
		int block_size;
		int vocab_size;
		std::string data;
		std::map<char, int> stoi;
		std::map<int, char> itos;
};

int main() {
	int block_size = 128;
	std::string text = open('input.txt').read();
	CharDataset train_dataset = CharDataset(text, block_size);
	for (int i = 0; i < train_dataset.size(); i++) {
		std::pair<torch::Tensor, torch::Tensor> xy = train_dataset.get(i);
		torch::Tensor x = xy.first;
		torch::Tensor y = xy.second;
		std::string str_x = "";
		std::string str_y = "";
		for (int j = 0; j < x.size()[0]; j++) {
			str_x += train_dataset.itos[x[j].item<int>()];
			str_y += train_dataset.itos[y[j].item<int>()];
		}
		std::cout << str_x << std::endl;
		std::cout << "_______________________________" << std::endl;
		std::cout << str_y << std::endl;
		break;
	}
	return 0;
}