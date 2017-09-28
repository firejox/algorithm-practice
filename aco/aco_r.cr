class Synapse
  property weight = 0.0
  property source_neuron : Neuron
  property dest_neuron : Neuron

  def initialize(@source_neuron, @dest_neuron)
  end
end

class Neuron
  property synapses_in = [] of Synapse
  property synapses_out = [] of Synapse

  property threshold = 0.0

  property output = 0.0

  def initialize
  end

  def calculate_output
    activation = synapses_in.reduce(-@threshold) do |sum, synapse|
      sum + synapse.weight * synapse.source_neuron.output
    end

    @output = 1.0 / (1.0 + Math.exp(-activation))
  end
end

# Full Connected Deep Neural Network
#
class FCDNeuralNetwork
  @input_layer : Array(Neuron)
  @hidden_layers : Array(Array(Neuron))
  @output_layer : Array(Neuron)
  @args_num = 0

  def initialize(input_num, hidden_num, output_num, depth = 1)
    @input_layer = Array.new(input_num) { Neuron.new }
    @hidden_layers = Array.new(depth) { Array.new(hidden_num) { Neuron.new } }
    @output_layer = Array.new(output_num) { Neuron.new }

    @args_num = hidden_num * (input_num + 1) + output_num * (hidden_num + 1) + hidden_num * (hidden_num + 1) * (depth - 1)

    @input_layer.product(@hidden_layers[0]) do |source, dest|
      synapse = Synapse.new(source, dest)
      source.synapses_out << synapse
      dest.synapses_in << synapse
    end

    if depth > 1
      1.upto(depth-1) do |i|
        @hidden_layers[i - 1].product(@hidden_layers[i]) do |source, dest|
          synapse = Synapse.new(source, dest)
          source.synapses_out << synapse
          dest.synapses_in << synapse
        end
      end
    end
    
    @hidden_layers[depth - 1].product(@output_layer) do |source, dest|
      synapse = Synapse.new(source, dest)
      source.synapses_out << synapse
      dest.synapses_in << synapse
    end
  end

  def args_num
    @args_num
  end

  def feed_forward(inputs)
    @input_layer.zip(inputs) do |neuron, input|
      neuron.output = input.to_f64
    end

    @hidden_layers.each do |hidden_layer|
      hidden_layer.each do |neuron|
        neuron.calculate_output
      end
    end

    @output_layer.each do |neuron|
      neuron.calculate_output
    end
  end

  def output
    @output_layer.map do |neuron|
      neuron.output
    end
  end

  def update_parameters(&block : Float64 -> Float64)
    @hidden_layers.each do |hidden_layer|
      hidden_layer.each do |neuron|
        neuron.synapses_in.each do |synapse|
          synapse.weight = yield synapse.weight
        end

        neuron.threshold = yield neuron.threshold
      end
    end

    @output_layer.each do |neuron|
      neuron.synapses_in.each do |synapse|
        synapse.weight = yield synapse.weight
      end

      neuron.threshold = yield neuron.threshold
    end
  end
end

class Ant
  property route : Array(Float64)
  property length = Float64::MAX

  def initialize(food_num : Int32, &block)
    @route = Array.new(food_num) { |i| yield i }
  end

  def dup
    ant = Ant.new(@route.size) { |i| @route[i] }
    ant.length = @length
    ant
  end
end

class ACONeuralNetwork
  Locality = 1e-4
  ConvergenceSpeed = 0.85

  @solution_size = 0
  @ants : Array(Ant)
  @prob_table : Array(Float64)

  def initialize(ant_num, @neural_network : FCDNeuralNetwork)
    @random_gen = Random.new

    food_num = @neural_network.args_num
    
    @solution_size = Math.max(food_num, 10)

    @ants = Array.new(ant_num + @solution_size) do |i|
      if i < @solution_size
        Ant.new(food_num) { @random_gen.rand(-1.0..1.0) }
      else
        Ant.new(food_num) { 0.0 }
      end
    end

    qk = Locality * @solution_size
    @prob_table = Array.new(@solution_size) { |i| Math.exp(-0.5 * (i * i) / (qk * qk)) / (qk * Math.sqrt(2.0 * Math::PI)) }

    total_prob = @prob_table.reduce(0.0) { |sum, prob| sum + prob }

    @prob_table.map! do |prob|
      prob / total_prob
    end

    1.upto(@solution_size - 1) do |i|
      @prob_table[i] += @prob_table[i-1]
    end
  end

  def solution_construction(input, output)
    ant_iter = @ants.each
    @solution_size.times do
      ant = ant_iter.next.as(Ant)
      visited_food = ant.route.each

      @neural_network.update_parameters do
        visited_food.next.as(Float64)
      end

      @neural_network.feed_forward(input)

      total_diff = output.zip(@neural_network.output).reduce(0.0) do |sum, ab|
        diff = ab[0] - ab[1]
        sum + diff * diff
      end

      ant.length = total_diff
    end

    ant_iter.each do |ant|
      ant.length = Float64::MAX
    end

    ant_iter.rewind

    @ants.sort_by! { |ant| ant.length }

    ant_iter.skip(@solution_size).each do |ant|
      ant.route.fill do |i|
        selected = @random_gen.rand
        mu = @ants[@prob_table.bsearch_index { |x, j| x >= selected }.not_nil!].route[i]

        sigma = ConvergenceSpeed * (@ants.each.first(@solution_size).map &.route[i]).reduce(0.0) do |sum, x|
          sum + (x - mu).abs
        end / (@solution_size - 1)

        @random_gen.normal(mu, sigma)
      end

      visited_food = ant.route.each

      @neural_network.update_parameters do
        visited_food.next.as(Float64)
      end

      @neural_network.feed_forward(input)

      total_diff = output.zip(@neural_network.output).reduce(0.0) do |sum, ab|
        diff = ab[0] - ab[1]
        sum + diff * diff
      end

      ant.length = total_diff
    end
  end

  def update_pheromones
    @ants.sort_by! { |ant| ant.length }
  end

  def optimal_route
    @ants[0].route.dup
  end 
end

module Random
  def normal(mu = 0.0, sigma = 1.0)
    normal2(mu, sigma)[0]
  end

  # Box-Muller method
  def normal2(mu = 0.0, sigma = 1.0)
    u = Math.sqrt(-2.0 * Math.log(rand))
    v = rand(Math::PI * 2.0)

    return (u * Math.cos(v) * sigma + mu), (u * Math.sin(v) * sigma + mu)
  end
end

identity_fn = FCDNeuralNetwork.new(1, 1, 1)
aco_nn = ACONeuralNetwork.new(2, identity_fn)

test_cases = [] of Tuple(Array(Float64), Array(Float64))


File.each_line("identity_samples") do |line|
  ary = line.split(' ').map { |s| s.to_f64 }

  test_cases << {[ary[0]], [ary[1]]}
end

1000_000.times do |i|
  test_cases.each do |test_case|
    aco_nn.solution_construction(test_case[0], test_case[1])
    aco_nn.update_pheromones
  end
end

best_visited_food = aco_nn.optimal_route.each
identity_fn.update_parameters do
  best_visited_food.next.as(Float64)
end

print "input: "
while str = gets
  identity_fn.feed_forward([str.to_f64])
  puts identity_fn.output
  print "input: "
end
