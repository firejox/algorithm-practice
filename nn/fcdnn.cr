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
