//! Module containting the required evolutionary NN implementation for the LSTMs

use rand::Rng;
use rand::distributions::{Normal, IndependentSample};


/// Specifies the activation function
#[derive(Debug, Copy, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum Activation
{
	/// Sigmoid activation
	Sigmoid,
	/// Tanh activation
	Tanh,
}

/// Evolutionary Neural Network for LSTM
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ENN
{
	generation: u32, //generation of current network
	num_inputs: u32, //number of inputs to NN
	num_hidden: u32, //size of every hidden layer
	num_outputs: u32, //number of outputs of the NN
	hidden_layers: u32, //number of hidden layers
	hid_act: Activation, //hidden layer activation
	out_act: Activation, //output layer activation
	layers: Vec<Vec<Vec<f64>>>, //NN layers -> nodes -> weights
}

#[allow(dead_code)]
impl ENN
{
	/// Creates a new neural net with the given parameters. Initially there is one hidden layer
	/// Be careful with Sigmoid as hidden layer activation function, as it could possibly slow down block additions
	pub fn new(inputs:u32, hidden_size:u32, outputs:u32, hidden_layers:u32, hidden_activation:Activation, output_activation:Activation) -> ENN
	{
		let mut rng = ::rand::thread_rng();
		if inputs < 1 || outputs < 1 || (hidden_layers > 0 && hidden_size < 1)
		{
			panic!("Inappropriate parameter bounds!");
		}
		
		// setup the layers
		let mut layers = Vec::new();
		let mut prev_layer_size = inputs;
		for i in 0..(1 + hidden_layers)
		{ //one hidden layer and one output layer
			let mut layer: Vec<Vec<f64>> = Vec::new();
			let layer_size = if i == hidden_layers { outputs } else { hidden_size };
			let normal = Normal::new(0.0, (2.0 / prev_layer_size as f64).sqrt()); //He init
			for _ in 0..layer_size
			{
				let mut node: Vec<f64> = Vec::with_capacity(1 + prev_layer_size as usize);
				for i in 0..prev_layer_size+1
				{
					if i == 0 //threshold aka bias
					{
						node.push(0.0);
					}
					else
					{
						let random_weight:f64 = normal.ind_sample(&mut rng);
						node.push(random_weight);
					}
				}
				layer.push(node);
			}
			layer.shrink_to_fit();
			layers.push(layer);
			prev_layer_size = layer_size;
		}
		layers.shrink_to_fit();
		
		//set activation functions
		ENN { generation: 0, num_inputs: inputs, num_hidden: hidden_size, num_outputs: outputs, hidden_layers: hidden_layers,
				hid_act: hidden_activation, out_act: output_activation, layers: layers }
	}
	
	/// Encodes the network as a JSON string.
	pub fn to_json(&self) -> String
	{
		::serde_json::to_string(self).ok().expect("Encoding JSON failed!")
	}

	/// Builds a new network from a JSON string.
	pub fn from_json(encoded:&str) -> ENN
	{
		let network:ENN = ::serde_json::from_str(encoded).ok().expect("Decoding JSON failed!");
		network
	}
	
	///Runs a feed-forward run through the network and returns the results (of the output layer)
	pub fn run(&self, inputs:&[f64]) -> Vec<f64>
	{
		if inputs.len() as u32 != self.num_inputs
		{
			panic!("Input has a different length than the network's input layer!");
		}
		
		self.do_run(inputs, 0.0).pop().unwrap()
	}
	
	///Runs a feed-forward run through the network using random drop-out and returns the results (of the output layer)
	///d = probability to drop the node respectively (must be in 0-1)
	pub fn run_dropout(&self, inputs:&[f64], d:f64) -> Vec<f64>
	{
		if inputs.len() as u32 != self.num_inputs
		{
			panic!("Input has a different length than the network's input layer!");
		}
		if d < 0.0 || d > 1.0
		{
			panic!("Probability to drop nodes not in correct range! [0.0, 1.0]");
		}
		
		self.do_run(inputs, d).pop().unwrap()
	}

	fn do_run(&self, inputs:&[f64], d:f64) -> Vec<Vec<f64>>
	{
		let mut rng = ::rand::thread_rng();
		let mut results = Vec::new();
		results.push(inputs.to_vec());
		for (layer_index, layer) in self.layers.iter().enumerate()
		{
			let mut layer_results = Vec::new();
			for node in layer.iter()
			{
				if d > 0.0 && rng.gen::<f64>() < d
				{ //drop node => result 0 so it has no effect
					layer_results.push(0.0);
				}
				else
				{ //keep node => calculate results
					let mut sum = modified_dotprod(&node, &results[layer_index]); //sum of forward pass to this node
					//standard forward pass activation
					let act;
					if layer_index == self.layers.len() - 1 //output layer
					{ act = self.out_act; }
					else { act = self.hid_act; }
					sum = match act {
								Activation::Sigmoid => sigmoid(sum),
								Activation::Tanh => tanh(sum),
							};
					//push result
					layer_results.push(sum);
				}
			}
			results.push(layer_results);
		}
		results
	}

	fn get_layers_mut(&mut self) -> &mut Vec<Vec<Vec<f64>>>
	{
		&mut self.layers
	}

	fn get_layers(&self) -> &Vec<Vec<Vec<f64>>>
	{
		&self.layers
	}

	pub fn get_gen(&self) -> u32
	{
		self.generation
	}

	fn set_gen(&mut self, gen:u32)
	{
		self.generation = gen;
	}
	
	pub fn get_num_hidden(&self) -> u32
	{
		self.num_hidden
	}

	///  breed a child from the 2 networks, either by random select or by averaging weights
	/// panics if the neural net's num_hidden are not the same
	pub fn breed(&self, other:&ENN, prob_avg:f64) -> ENN
	{
		if self.num_hidden != other.num_hidden
		{
			panic!("Incompatible networks to breed!");
		}
		
		let mut rng = ::rand::thread_rng();
		let mut newnn = self.clone();
		
		//set generation
		let oldgen = newnn.get_gen();
		newnn.set_gen((other.get_gen() + oldgen + 3) / 2); //+ 1 and round up
		
		//activation functions are kept
		
		//set parameters
		{ //put in scope, because of mutable borrow before ownership return
			let layers1 = newnn.get_layers_mut();
			let layers1len = layers1.len();
			let layers2 = other.get_layers();
			for layer_index in 0..layers1len
			{
				let layer = &mut layers1[layer_index];
				for node_index in 0..layer.len()
				{
					let node = &mut layer[node_index];
					for weight_index in 0..node.len()
					{
						let mut layer2val = 0.0;
						if layer_index == layers1len - 1 //last layer
						{ //use the same layer weights again for the output layer, also if network 2 is deeper
							let outlayer_i = layers2.len() - 1;
							layer2val = layers2[outlayer_i][node_index][weight_index];
						}
						else if layer_index < layers2.len() - 1
						{ //simulate same network size by using zeros for the block
							layer2val = layers2[layer_index][node_index][weight_index];
						} //if layers2 is deeper than layers1, the shorter layers1 is taken and deeper layers ignored
						
						if prob_avg == 1.0 || (prob_avg != 0.0 && rng.gen::<f64>() < prob_avg)
						{ //average between weights
							node[weight_index] = (node[weight_index] + layer2val) / 2.0;
						}
						else
						{
							if rng.gen::<f64>() < 0.5
							{ //random if stay at current weight or take father's/mother's
								node[weight_index] = layer2val;
							}
						}
					}
				}
			}
		}
		
		//return
		newnn
	}

	/// mutate the current network
	/// params: (all probabilities in [0,1])
	/// prob_new:f64 - probability to become a new freshly initialized network of same size/architecture (to change hidden size create one manually and don't breed them)
	/// prob_op:f64 - probability to apply an operation to a weight
	/// op_range:f64 - maximum absolute adjustment of a weight
	pub fn mutate(&mut self, prob_new:f64, prob_op:f64, op_range:f64)
	{
		let mut rng = ::rand::thread_rng();
		//fresh random network parameters
		if rng.gen::<f64>() < prob_new
		{
			self.mutate_new();
		}
		//random addition / substraction op mutation
		if prob_op != 0.0 && op_range != 0.0
		{
			self.mutate_op(prob_op, op_range);
		}
	}

	fn mutate_new(&mut self)
	{
		let mut rng = ::rand::thread_rng();
		let mut prev_layer_size = self.num_inputs as usize;
		for layer_index in 0..self.layers.len()
		{
			let layer = &mut self.layers[layer_index];
			let normal = Normal::new(0.0, (2.0 / prev_layer_size as f64).sqrt()); //HE init
			for node_index in 0..layer.len()
			{
				let node = &mut layer[node_index];
				for weight_index in 0..node.len()
				{
					node[weight_index] = if weight_index == 0 { 0.0 } else { normal.ind_sample(&mut rng) };
				}
			}
			prev_layer_size = layer.len();
		}
	}

	/// mutate using addition/substraction of a random value (random per node)
	fn mutate_op(&mut self, prob_op:f64, op_range:f64)
	{
		let mut rng = ::rand::thread_rng();
		for layer_index in 0..self.layers.len()
		{
			let layer = &mut self.layers[layer_index];
			for node_index in 0..layer.len()
			{
				let node = &mut layer[node_index];
				for weight_index in 0..node.len()
				{
					//possibly modify weight
					if rng.gen::<f64>() < prob_op
					{ //RNG says this weight will be changed
						let delta = op_range * (2.0 * rng.gen::<f64>() - 1.0);
						node[weight_index] += delta;
					}
				}
			}
		}
	}
}

///sigmoid
fn sigmoid(x:f64) -> f64
{
    1f64 / (1f64 + (-x).exp())
}

///tanh
fn tanh(x:f64) -> f64
{
	x.tanh()
	//2.0 * sigmoid(x) - 1.0
}


fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64
{
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); //start with the threshold weight
    for (weight, value) in it.zip(values.iter())
	{
        total += weight * value;
    }
    total
}

