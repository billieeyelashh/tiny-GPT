#include <iostream> 
#include <math.h> 
#include <logging> 
#include <torch/torch.h>
#include <nn.h>

class GPTConfig {
public: 
	float embd_pdrop;
	float resid_pdrop;
	float attn_pdrop;

	int vocab_size;
	int block_size;

	GPTConfig(int vocab_size_, int block_size_, **kwargs) {
		this->vocab_size = vocab_size_;
		this->block_size = block_size_;
		for (auto k : kwargs) {
			setattr(this, k, v);
		}
	}
};

class GPT1Config : public GPTConfig {
public:
	int n_layer;
	int n_head;
	int n_embd;

	GPT1Config(int vocab_size_, int block_size_, **kwargs) : GPTConfig(vocab_size_, block_size_, **kwargs) {
		this->n_layer = 12;
		this->n_head = 12;
		this->n_embd = 768;
	}
};

class CasualSelfAttention : public torch::nn::Module {
public:
	torch::Tensor key;
	torch::Tensor query;
	torch::Tensor value;

	torch::nn::Dropout attn_drop;
	torch::nn::Dropout resid_drop;

	torch::Tensor proj;

	torch::Tensor mask;
	int n_head;

	CasualSelfAttention(GPTConfig config) {
		assert(config.n_embd % config.n_head == 0);

		this->key = torch::nn::Linear(config.n_embd, config.n_embd);
		this->query = torch::nn::Linear(config.n_embd, config.n_embd);
		this->value = torch::nn::Linear(config.n_embd, config.n_embd);

		this->attn_drop = torch::nn::Dropout(config.attn_pdrop);
		this->resid_drop = torch::nn::Dropout(config.resid_pdrop);

		this->proj = torch::nn::Linear(config.n_embd, config.n_embd);

		torch::Tensor tri_ones = torch::tril(torch::ones(config.block_size, config.block_size));
		this->mask = tri_ones.view({1, 1, config.block_size, config.block_size});
		this->n_head = config.n_head;
	}

	torch::Tensor forward(torch::Tensor x, torch::Tensor layer_past=nullptr) {
		int B, T, C; 
		std::tie(B, T, C) = x.size();

		torch::Tensor k = this->key(x).view({B, T, this->n_head, C / this->n_head}).transpose(1, 2);
		torch::Tensor q = this->query(x).view({B, T, this->n_head, C / this->n_head}).transpose(1, 2);
		torch::Tensor v = this->value(x).view({B, T, this->n_head, C / this->n_head}).transpose(1, 2);

		torch::Tensor att = (q @ k.transpose(-2, -1)) * (1.0 / sqrt(k.size(-1)));
		att = att.masked_fill(this->mask[{{}, {}, {0, T}, {0, T}}] == 0, float('-inf'));
		att = torch::nn::functional::softmax(att, -1);
		att = this->attn_drop(att);
		torch::Tensor y = att @ v;
		y = y.transpose(1, 2).contiguous().view({B, T, C});

		y = this->resid_drop(this->proj(y));
		return y;
	}
};

class Block : public torch::nn::Module {
public:
	torch::nn::LayerNorm ln1;
	torch::nn::LayerNorm ln2;
	CasualSelfAttention attn;
	torch::nn::Sequential mlp;

	Block(GPTConfig config) {
		this->ln1 = torch::nn::LayerNorm(config.n_embd);
		this->ln2 = torch::nn::LayerNorm(config.n_embd);
		this->attn = CasualSelfAttention(config);
		this->mlp = torch::nn::Sequential(torch::nn::Linear(config.n_embd, 4*config.n_embd),
										 torch::nn::GELU(),
										 torch::nn::Linear(4*config.n_embd, config.n_embd),
										 torch::nn::Dropout(config.resid_pdrop));
	}

	torch::Tensor forward(torch::Tensor x) {
		x = x + this->attn(this->ln1(x));
		x = x + this->mlp(this->ln1(x));
		return x;
	}
};

class GPT
{
public:
	GPT(config)
	{
		tok_embd = nn.Embedding(config.vocab_size, config.n_embd);
		pos_embd = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd));
		drop = nn.Dropout(config.embd_pdrop);

		blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]);

		ln_f = nn.LayerNorm(config.n_embd);
		head = nn.Linear(config.n_embd, config.vocab_size, bias=False);

		block_size = config.block_size;
		this->__init_weights();

		logger.info(f'Number of parameters: {sum(p.numel() for p in this->parameters())}');
	}

	int get_block_size()
	{
		return block_size;
	}

	void __init_weights(module)
	{
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02);
			if isinstance(module, nn.Linear) and (not (module.bias is None)):
				module.bias.data.zero_();
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_();
			module.weight.data.fill_(1.0);
	}

	void configure_optimizers(train_config)
	{
		decay = set();
		no_decay = set();
		whitelist_weight_modules = (torch.nn.Linear, );
		blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding);
		for mn, m in this->named_modules():
			for pn, p in m.named_parameters():
				fpn = f"{mn}.{pn}" if mn else pn;
				if pn.endswith('bias'):
					no_decay.add(fpn);
				elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
					decay.add(fpn);
				elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
					no_decay.add(fpn);
		no_decay.add('pos_embd');

		param_dict = {pn: p for pn, p in this->named_parameters()};
		inter_params = decay & no_decay;
		union_params = decay | no_decay;
		assert len(inter_params) == 0, f"Parameters {str(inter_params)} made into two dicts";
		assert len(union_params - param_dict.keys()) == 0, f"Parameters {str(param_dict.keys() - union_params)} were ommited";

		optim_groups = [
			{"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
			{"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
		];
		optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas);
		return optimizer;
	}

	forward(idx, targets=None)
	{
		b, t = idx.size();
		assert t <= this->block_size, "Block size exhausted";

		token_embeddings = this->tok_embd(idx);
		position_embeddings = this->pos_embd[:, :t, :];
		x = this->drop(token_embeddings + position_embeddings);
		x = this->blocks(x);
		x = this->ln_f(x);
		logits = this->head(x);

		loss = NULL;
		if not (targets is None):
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1));

		return logits, loss;
	}

private:
	nn.Embedding tok_embd;
	nn.Parameter pos_embd;
	nn.Dropout drop;
	nn.Sequential blocks;
	nn.LayerNorm ln_f;
	nn.Linear head;
	int block_size;
};
