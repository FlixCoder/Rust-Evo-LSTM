//! Evolutionary LSTMs by FlixCoder

#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate rand;
extern crate rayon;
mod networks;

use rand::Rng;
use rayon::prelude::*;
use networks::*;
use std::cmp::Ordering;


/// Trait to define evaluators in order to use the algorithm in a flexible way
pub trait LSTMEvaluator
{
	///evaluate LSTM (USE ONLY RUN)
	fn evaluate(&self, lstm:&mut LSTM) -> f64; //returns rating of LSTM (higher is better (you can inverse with -))
}

/// Optimizer class to optimize LSTMs by evolutionary / genetic algorithms
/// For parallel optimization the evaluator has to implement the Sync-trait! Regarding controlling the number of threads, see Rayon's documentation
pub struct Optimizer<T:LSTMEvaluator>
{
	eval: T, //evaluator
	nets: Vec<(LSTM, f64)>, //population of LSTMs and ratings (sorted, high/best rating in front)
}

impl<T:LSTMEvaluator> Optimizer<T>
{
	/// Create a new optimizer using the given evaluator for the given neural net
	pub fn new(evaluator:T, mut lstm:LSTM) -> Optimizer<T>
	{
		let mut netvec = Vec::new();
		let rating = evaluator.evaluate(&mut lstm);
		netvec.push((lstm, rating));
		
		Optimizer { eval: evaluator, nets: netvec }
	}
	
	/// Get a reference to the population
	pub fn get_population(&self) -> &Vec<(LSTM, f64)>
	{
		&self.nets
	}
	
	/// Get a mutable reference to the population (there is no set_population, use this)
	pub fn get_population_mut(&mut self) -> &mut Vec<(LSTM, f64)>
	{
		&mut self.nets
	}
	
	/// Save population as json string and return it
	pub fn save_population(&self) -> String
	{
		serde_json::to_string(&self.nets).ok().expect("Encoding JSON failed!")
	}
	
	/// Load population from json string
	pub fn load_population(&mut self, encoded:&str)
	{
		self.nets = serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
	}
	
	/// Get a reference to the used evaluator
	pub fn get_eval(&self) -> &T
	{
		&self.eval
	}
	
	/// Switch to a new evaluator to allow change of evaluation. you should probably call reevaluate afterwards
	pub fn set_eval(&mut self, evaluator:T)
	{
		self.eval = evaluator;
	}
	
	/// Clones the best LSTM an returns it
	pub fn get_lstm(&mut self) -> LSTM
	{
		self.nets[0].0.clone()
	}
	
	/// Reevaluates all neural nets based on the current (possibly changed) evaluator, returns best score
	pub fn reevaluate(&mut self) -> f64
	{
		//evaluate
		for net in &mut self.nets
		{
			net.1 = self.eval.evaluate(&mut net.0);
		}
		//sort and return
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Same as reevaluate but in parallel
	pub fn reevaluate_par(&mut self) -> f64
		where T:Sync
	{
		//evaluate
		{ //scope for borrowing
			let eval = &self.eval;
			self.nets.par_iter_mut().for_each(|net|
				{
					net.1 = eval.evaluate(&mut net.0);
				});
		}
		//sort and return
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Optimizes the LSTMs for the given number of generations.
	/// Returns the rating of the best LSTM score afterwards.
	/// 
	/// Parameters: (probabilities are in [0,1]) (see xor example in ERNN for help regarding paramter choice)
	/// generations - number of generations to optimize over
	/// population - size of population to grow up to
	/// survival - number nets to survive by best rating
	/// bad_survival - number of nets to survive randomly from nets, that are not already selected to survive from best rating
	/// prob_avg - probability to use average weight instead of selection in breeding
	/// prob_mut - probability to mutate after breed
	/// prob_new - probability to generate a new random network
	/// prob_op - probability for each weight to mutate using an delta math operation during mutation
	/// op_range - factor to control the range in which delta can be in
	pub fn optimize(&mut self, generations:u32, population:u32, survival:u32, bad_survival:u32, prob_avg:f64, prob_mut:f64,
					prob_new:f64, prob_op:f64, op_range:f64) -> f64
	{
		//optimize for "generations" generations
		for _ in 0..generations
		{
			let children = self.populate(population as usize, prob_avg, prob_mut, prob_new, prob_op, op_range);
			self.evaluate(children);
			self.sort_nets();
			self.survive(survival, bad_survival);
			//self.sort_nets(); //not needed, because population generation is choosing randomly
		}
		//return best rating
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Same as optimize, but in parallel.
	pub fn optimize_par(&mut self, generations:u32, population:u32, survival:u32, bad_survival:u32, prob_avg:f64, prob_mut:f64,
					prob_new:f64, prob_op:f64, op_range:f64) -> f64
		where T:Sync
	{
		//optimize for "generations" generations
		for _ in 0..generations
		{
			let children = self.populate(population as usize, prob_avg, prob_mut, prob_new, prob_op, op_range);
			self.evaluate_par(children);
			self.sort_nets();
			self.survive(survival, bad_survival);
			//self.sort_nets(); //not needed, because population generation is choosing randomly!
		}
		//return best rating
		self.sort_nets();
		self.nets[0].1
	}
	
	/// Easy shortcut to optimize using standard parameters.
	/// For paramters see optimize.
	pub fn optimize_easy(&mut self, generations:u32, population:u32, prob_op:f64, op_range:f64) -> f64
	{
		//standard parameter choice
		let survival = 4;
		let badsurv = 1;
		let prob_avg = 0.1;
		let prob_mut = 0.95;
		let prob_new = 0.1;
		self.optimize(generations, population, survival, badsurv, prob_avg, prob_mut, prob_new, prob_op, op_range)
	}
	
	/// Same as optimize_easy, but in parallel
	pub fn optimize_easy_par(&mut self, generations:u32, population:u32, prob_op:f64, op_range:f64) -> f64
		where T:Sync
	{
		//standard parameter choice
		let survival = 4;
		let badsurv = 1;
		let prob_avg = 0.1;
		let prob_mut = 0.95;
		let prob_new = 0.1;
		self.optimize_par(generations, population, survival, badsurv, prob_avg, prob_mut, prob_new, prob_op, op_range)
	}
	
	/// Generate initial random population of the given size.
	/// Just a shortcut to optimize with less parameters.
	pub fn gen_population(&mut self, population:u32) -> f64
	{
		self.optimize(1, population, population, 0, 0.0, 1.0, 1.0, 0.0, 0.0)
	}
	
	/// Generates new population and returns a vec of nets, that need to be evaluated
	fn populate(&self, size:usize, prob_avg:f64, prob_mut:f64, prob_new:f64, prob_op:f64, op_range:f64) -> Vec<(LSTM, f64)>
	{
		let mut rng = rand::thread_rng();
		let len = self.nets.len();
		let missing = size - len;
		let mut newpop = Vec::new();
		
		for _ in 0..missing
		{
			let i1:usize = rng.gen::<usize>() % len;
			let i2:usize = rng.gen::<usize>() % len;
			let othernn = &self.nets[i2].0;
			let mut newnn = self.nets[i1].0.breed(othernn, prob_avg);
			
			if rng.gen::<f64>() < prob_mut
			{
				newnn.mutate(prob_new, prob_op, op_range);
			}
			
			newpop.push((newnn, 0.0));
		}
		
		newpop
	}
	
	/// Evaluates a given set of NNs and appends them into the internal storage
	fn evaluate(&mut self, mut nets:Vec<(LSTM, f64)>)
	{
		for net in &mut nets
		{
			net.1 = self.eval.evaluate(&mut net.0);
		}
		self.nets.append(&mut nets);
	}
	
	/// Same as evaluate but in parallel
	fn evaluate_par(&mut self, mut nets:Vec<(LSTM, f64)>)
		where T:Sync
	{
		nets.par_iter_mut().for_each(|net|
			{
				net.1 = self.eval.evaluate(&mut net.0);
			});
		self.nets.append(&mut nets);
	}
	
	/// Eliminates population, so that the best "survival" nets and random "bad_survival" nets survive
	fn survive(&mut self, survival:u32, bad_survival:u32)
	{
		if survival as usize >= self.nets.len() { return; } //already done
		
		let mut rng = rand::thread_rng();
		let mut bad = self.nets.split_off(survival as usize);
		
		for _ in 0..bad_survival
		{
			if bad.is_empty() { return; }
			let i:usize = rng.gen::<usize>() % bad.len();
			self.nets.push(bad.swap_remove(i));
		}
	}
	
	/// Sorts the internal NNs, so that the best net is in front (index 0)
	fn sort_nets(&mut self)
	{ //best nets (high score) in front, bad and NaN nets at the end
		self.nets.sort_by(|ref r1, ref r2| { //reverse partial cmp and check for NaN
				let r = (r2.1).partial_cmp(&r1.1);
				if r.is_some() { r.unwrap() }
				else
				{
					if r1.1.is_nan() { if r2.1.is_nan() { Ordering::Equal } else { Ordering::Greater } } else { Ordering::Less }
				}
			});
	}
}



///LSTM-like structured architecture
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct LSTM
{
	generation: u32, //generation of current network
	num_inputs: u32, //input dimension
	num_mem: u32, //memory dimension
	num_outputs: u32, //output dimension
	cur_mem: Vec<f64>, //current memory vector
	cur_out: Vec<f64>, //current output vector
	gate_forget: ENN, //ENN to gate forgetting and, after inversion, information gain
	net_extract: ENN, //ENN to process old output, memory and input into new information
	net_output: ENN, //ENN to process input, old output and memory into new output
}

impl LSTM
{
	///Create a new simple LSTM
	/// inputs = dimension of input data
	/// mem = dimension of memory vector
	/// outputs = dimension of output data
	pub fn new(inputs:u32, mem:u32, outputs:u32) -> LSTM
	{
		LSTM::new_ex(inputs, mem, outputs, 0, 0)
	}
	
	///Create a new simple LSTM
	/// inputs = dimension of input data
	/// mem = dimension of memory vector
	/// outputs = dimension of output data
	/// hiddens = number of hidden layers (for all networks)
	/// num_hidden = size of hidden all layers
	pub fn new_ex(inputs:u32, mem:u32, outputs:u32, hiddens:u32, num_hidden:u32) -> LSTM
	{
		let memvec = vec![0.0; mem as usize];
		let outvec = vec![0.0; outputs as usize];
		let nn_in_size = inputs + mem + outputs;
		let forget = ENN::new(nn_in_size, num_hidden, mem, hiddens, Activation::Tanh, Activation::Sigmoid);
		let extract = ENN::new(nn_in_size, num_hidden, mem, hiddens, Activation::Tanh, Activation::Tanh);
		let output = ENN::new(nn_in_size, num_hidden, outputs, hiddens, Activation::Tanh, Activation::Tanh);
		
		LSTM { generation: 0, num_inputs: inputs, num_mem: mem, num_outputs: outputs,
				cur_mem: memvec, cur_out: outvec,
				gate_forget: forget, net_extract: extract, net_output: output }
	}
	
	///Encodes the LSTM as a JSON string.
	pub fn to_json(&self) -> String
	{
		serde_json::to_string(self).ok().expect("Encoding JSON failed!")
	}

	///Builds a new LSTM from a JSON string.
	pub fn from_json(encoded:&str) -> LSTM
	{
		let network:LSTM = serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
		network
	}
	
	///Reset current memory and output
	pub fn reset(&mut self)
	{
		self.cur_mem = vec![0.0; self.num_mem as usize];
		self.cur_out = vec![0.0; self.num_outputs as usize];
	}
	
	///Runs an LSTM iteration
	pub fn run(&mut self, inputs:&[f64]) -> Vec<f64>
	{
		if inputs.len() as u32 != self.num_inputs
		{
			panic!("Input dimension incorrect for this LSTM!");
		}
		
		//vector of input and old output
		let mut in_out = self.cur_out.clone();
		in_out.extend_from_slice(inputs);
		//adding memory is input for gate_forget and net_extract
		let mut nn_in = in_out.clone();
		nn_in.extend_from_slice(&self.cur_mem);
		
		//calculate memory update
		let mut gate = self.gate_forget.run(&nn_in);
		let mut info = self.net_extract.run(&nn_in);
		elem_multiply(&mut self.cur_mem, &gate);
		elem_invert(&mut gate);
		elem_multiply(&mut info, &gate);
		elem_add(&mut self.cur_mem, &info);
		
		//adding new memory is input for net_output
		in_out.extend_from_slice(&self.cur_mem);
		
		//calculate output
		self.cur_out = self.net_output.run(&in_out);
		self.cur_out.clone()
	}
	
	///Get LSTM's generation
	pub fn get_gen(&self) -> u32
	{
		self.generation
	}
	
	///Set LSTM's generation
	fn set_gen(&mut self, gen:u32)
	{
		self.generation = gen;
	}
	
	///Breeds a new LSTM from itself and another LSTM
	///prob_avg = probability to calculate average of weights instead of selection [0.0, 1.0]
	pub fn breed(&self, other:&LSTM, prob_avg:f64) -> LSTM
	{
		if self.num_inputs != other.num_inputs || self.num_mem != other.num_mem || self.num_outputs != other.num_outputs
			|| self.gate_forget.get_num_hidden() != other.gate_forget.get_num_hidden()
		{
			panic!("The LSTM's parameters do not fit together, can not breed!");
		}
		
		let mut lstm = self.clone();
		
		//set generation
		lstm.set_gen((other.get_gen() + self.get_gen() + 3) / 2); //+ 1 and round up
		
		//breed networks
		lstm.gate_forget = self.gate_forget.breed(&other.gate_forget, prob_avg);
		lstm.net_extract = self.net_extract.breed(&other.net_extract, prob_avg);
		lstm.net_output = self.net_output.breed(&other.net_output, prob_avg);
		
		//return
		lstm
	}
	
	///Mutate the current LSTM
	///Params: (all probabilities in [0,1])
	/// prob_new - probability to reinitialize weights randomly
	/// prob_op - probability to apply an operation to a weight
	/// op_range - maximum absolute adjustment to a weight
	pub fn mutate(&mut self, prob_new:f64, prob_op:f64, op_range:f64)
	{
		self.gate_forget.mutate(prob_new, prob_op, op_range);
		self.net_extract.mutate(prob_new, prob_op, op_range);
		self.net_output.mutate(prob_new, prob_op, op_range);
	}
}


///Elementwise vector multiplication
fn elem_multiply(a:&mut[f64], b:&[f64])
{
	if a.len() != b.len()
	{
		panic!("Incorrect parameter dimensions!");
	}
	
	for i in 0..a.len()
	{
		a[i] *= b[i]
	}
}

///Elementwise vector "inversion"
fn elem_invert(a:&mut[f64])
{
	for val in a.iter_mut()
	{
		*val = 1.0 - *val;
	}
}

///Elementwise vector addition
fn elem_add(a:&mut[f64], b:&[f64])
{
	if a.len() != b.len()
	{
		panic!("Incorrect parameter dimensions!");
	}
	
	for i in 0..a.len()
	{
		a[i] += b[i]
	}
}

