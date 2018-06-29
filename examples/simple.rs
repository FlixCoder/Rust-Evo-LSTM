extern crate elstm;

use elstm::*;


fn main()
{
	//parameters for genetic optimization
	let population = 250; //the bigger the network, the bigger this. however, this affects performance very hard, but stabilizes learning
	let survival = 7; //probably keep this
	let badsurv = 3; //probably keep this
	let prob_avg = 0.1; //probably keep this, does not change very much
	let prob_mut = 0.95; //probably keep this, not too much change
	let prob_new = 0.1; //can be adjusted, but do not set to 0.0, does not change too much, but avoids getting stuck
	let prob_op = 0.1; //the bigger the network, the lower this (try op_range first)
	let op_range = 0.1; //the bigger the network, the lower this (lower this before prob_op)
	
    // create a new lstm, evaluator and optimizer
	let lstm = LSTM::new(1, 5, 1);
	let eval = SimpleEval::new();
	let mut opt = Optimizer::new(eval, lstm);
	//generate initial population
	let mut mse = -opt.gen_population(survival + badsurv);
	
    // train the network
	let mut i = 0;
    while mse > 0.01
	{
		mse = -opt.optimize(10, population, survival, badsurv, prob_avg, prob_mut, prob_new, prob_op, op_range);
		println!("MSE: {}", mse);
		i += 10;
	}
	
	let mut lstm = opt.get_lstm(); //get the best LSTM
	let eval = SimpleEval::new();
	eval.print_results(&mut lstm);
	println!("LSTM information:");
	println!("Generation/Iterations: {}/{}", lstm.get_gen(), i);
}


//Simple constructed LSTM problem: remembering numbers for 3 iterations
struct SimpleEval
{
	examples: Vec<Vec<(Vec<f64>, f64)>>,
}

impl SimpleEval
{
	fn new() -> SimpleEval
	{
		let sequences = vec![
			vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
			vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
		];
		//create examples from sequences
		let mut examples = Vec::new();
		for seq in sequences.iter()
		{
			let mut example = Vec::new();
			for i in 0..seq.len()+3
			{
				let input = vec![ if i < seq.len() { seq[i] - 0.5 } else { 0.0 } ];
				let output = if i >= 3 { seq[i - 3] - 0.5 } else { 0.0 };
				example.push((input, output));
			}
			examples.push(example);
		}
		
		SimpleEval{ examples: examples }
	}
	
	fn print_results(&self, lstm:&mut LSTM)
	{
		for example in self.examples.iter()
		{
			println!("Start:");
			for (inputs, output) in example.iter()
			{
				let results = lstm.run(inputs);
				println!("Input {:5.2}; Output {:5.2} | {:5.2}", inputs[0], results[0], output);
			}
			lstm.reset();
		}
	}
}

//implement an evaluator to rate the LSTMs' results
impl LSTMEvaluator for SimpleEval
{
	fn evaluate(&self, lstm:&mut LSTM) -> f64
	{ //optimize the mean squared error
		let mut sum = 0.0;
		for example in self.examples.iter()
		{
			for (inputs, output) in example.iter()
			{
				let results = lstm.run(inputs);
				let diff = results[0] - output;
				sum += diff * diff;
			}
			lstm.reset();
		}
		sum /= self.examples.len() as f64;
		-sum //higher is good, so return minus mean squared error
	}
}
