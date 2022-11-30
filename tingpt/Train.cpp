#include <iostream>
#include <math.h>
#include <logging>

#include <numpy>
#include <torch>
#include <torch/optim.h>
#include <torch/scheduler.h>
#include <torch/nn/functional.h>

class TrainerConfig
{
public:
	int max_epochs;
	int batch_size;
	float learning_rate;
	float betas[2];
	float grad_norm_clip;
	float weight_decay;
	bool lr_decay;
	float warmup_tokens;
	float final_tokens;
	std::string ckpt_path;
	int num_workers;

	TrainerConfig(**kwargs)
	{
		for (auto k, v : kwargs)
		{
			setattr(this, k, v);
		}
	}
};

class Trainer
{
public:
	torch::nn::Module model;
	TrainerConfig config;
	std::string device;

	std::unique_ptr<torch::optim::Optimizer> optimizer;
	int tokens;

	Trainer(torch::nn::Module model, TrainerConfig config, std::vector<int> device_ids=nullptr)
		: model(model), config(config)
	{
		device = "cpu";
		if (torch::cuda::is_available())
		{
			device = torch::cuda::current_device();
			if (!device_ids.empty())
			{
				model = torch::nn::DataParallel(model, device_ids).to(device);
			}
			else
			{
				model.to(device);
			}
		}

		torch::nn::Module raw_model = model.module() if (hasattr(model, "module")) else model;
		optimizer = raw_model.configure_optimizers(config);
		tokens = 0;
	}

	void save_checkpoint()
	{
		torch::nn::Module raw_model = model.module() if (hasattr(model, "module")) else model;
		logging.info(f'Saving {config.ckpt_path}');
		torch::save(raw_model.state_dict(), config.ckpt_path);
	}

	void load_checkpoint()
	{
		torch::nn::Module raw_model = model.module() if (hasattr(model, "module")) else model;
		logging.info(f'Loading {config.ckpt_path}');
		torch::load(config.ckpt_path);
		raw_model.load_state_dict(checkpoint);
	}

	float step(torch::Tensor x, torch::Tensor y)
	{
		model.train();
		x.to(device);
		y.to(device);

		std::tuple<torch::Tensor, torch::Tensor> logits_loss = model(x, y);
		torch::Tensor loss = std::get<1>(logits_loss).mean();
		model.zero_grad();
		loss.backward();
		torch::nn::utils::clip_grad_norm_(model.parameters(), config.grad_norm_clip);
		optimizer->step();

		if (config.lr_decay)
		{
			tokens += (y>=0).sum().item();
			if (tokens < config.warmup_tokens)
			{
				lr_mult = float(tokens)/float(std::max(1, config.warmup_tokens));
			}
			else
			{
				float progress = float(tokens - config.warmup_tokens)/float(std::max(1, config.final_tokens - config.warmup_tokens));
				lr_mult = std::max(0.1, 0.5*(1.0 + cos(M_PI*progress)));
			}
			float lr = config.learning_rate * lr_mult;
			for (auto param_group : optimizer->param_groups)
			{
				param_group['lr'] = lr;
			}
		}
		else
		{
			lr = config.learning_rate;
		}

		return loss.item();
	}

	torch::Tensor sample(torch::Tensor x, int steps, float temperature=1.0, bool sample=false)
	{
		model.eval();
		int block_size = model.get_block_size();
		for (int k = 0; k < steps; k++)
		{
			torch::Tensor x_cond = x if (x.size(1)<=block_size) else x[:, -block_size:];
			std::tuple<torch::Tensor, torch::Tensor> logits_loss = model(x_cond);
			torch::Tensor logits = std::get<0>(logits_loss)[:, -1, :]/temperature;
			torch::Tensor probs = torch::nn::functional::softmax(logits, dim=-1);
			torch::Tensor ix;
			if (sample)
			{
				ix = torch::multinomial(probs, num_samples=1);
			}
			else
			{
				std::tuple<torch::Tensor, torch::Tensor> _topk = torch::topk(probs, k=1, dim=-1);
				ix = std::get<1>(_topk);
			}
			x = torch::cat({x, ix}, dim=1);
		}
		return x;
	}
};