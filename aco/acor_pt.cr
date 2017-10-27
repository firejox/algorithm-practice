require "../nn/fcdnn"
require "./ant"

class ACORPTNeuralNetwork
  @neural_network : FCDNeuralNetwork
  @ants : Array(Ant)
  @weight_probs : Array(Float64)
  @solution_size : Int32

  @alpha : Float64
  @beta : Float64
  @gamma : Float64
  @lambda : Float64
  @conv_rate : Float64

  def initialize(**args)
    @rand_gen = Random.new
    ant_num = args[:ant_num]
    @neural_network = args[:ann]
    food_num = @neural_network.args_num
    @solution_size = Math.max(food_num, 10)

    @ants = Array.new(ant_num + @solution_size) do |i|
      if i < @solution_size
        Ant.new(food_num) { @rand_gen.rand(-1.0..1.0) }
      else
        Ant.new(food_num) { 0.0 }
      end
    end

    locality = args[:q] * @solution_size
    @conv_rate = args[:conv_rate]
    @gamma = args[:gamma]
    @alpha = args[:alpha]
    @beta = args[:beta]
    @lambda = args[:lambda]

    @weight_probs = Array.new(@solution_size) do |i|
      Math.exp(-0.5 * ((i / locality) ** 2))
    end

    total_w = @weight_probs.sum

    @weight_probs.map! do |w|
      w /= total_w

      (w ** @gamma)/((w ** @gamma + (1.0 - w) ** @gamma) ** (1.0 / @gamma))
    end
  end

  def solution_construction(input, output)
    ant_slice = Slice.new(@ants.to_unsafe, @ants.size)
    ant_slice[0, @solution_size - 1].each do |ant|
      visited_food = ant.route.each

      @neural_network.update_parameters do
        visited_food.next.as(Float64)
      end

      @neural_network.feed_forward(input)

      ant.length = output.zip(@neural_network.output).sum do |(a, b)|
        (a - b) ** 2
      end
    end

    (ant_slice + @solution_size).each { |ant| ant.length = Float64::MAX }
    
    @ants.sort_by! { |ant| ant.length }

    mean = ant_slice[0, @solution_size - 1].sum { |ant| ant.length } / @solution_size

    # max prospect
    selected = (0...@solution_size).max_by do |i|
      x = @ants[i].length - mean
      if x >= 0.0
        (x ** @alpha) * @weight_probs[i]
      else
        (-@lambda * ((-x) ** @beta)) * @weight_probs[i]
      end
    end

    @neural_network.args_num.times do |i|
      mu = @ants[selected].route[i]
      sigma = @conv_rate * ant_slice[0, @solution_size - 1].sum do |ant|
        (ant.route[i] - mu).abs
      end / (@solution_size - 1)

      (ant_slice + @solution_size).each do |ant|
        ant.route[i] = @rand_gen.normal mu, sigma
      end
    end

    (ant_slice + @solution_size).each do |ant|
      visited_food = ant.route.each

      @neural_network.update_parameters do
        visited_food.next.as(Float64)
      end

      @neural_network.feed_forward(input)

      ant.length = output.zip(@neural_network.output).sum do |(a, b)|
        (a - b) ** 2
      end
    end
  end

  def update_pheromones
    @ants.sort_by! { |ant| ant.length }
  end

  def optimal_route
    solutions_slice = Slice.new(@ants.to_unsafe, @solution_size)
    mean = solutions_slice.sum { |ant| ant.length } / @solution_size

    selected = (0...@solution_size).max_by do |i|
      x = solutions_slice[i].length - mean
      if x >= 0.0
        (x ** @alpha) * @weight_probs[i]
      else
        (-@lambda * ((-x) ** @beta)) * @weight_probs[i]
      end
    end

    @ants[selected].route.dup
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
aco_nn = ACORPTNeuralNetwork.new ant_num: 2, ann: identity_fn, q: 1e-3,
  alpha: 0.88, beta: 0.88, gamma: 0.68, lambda: 2.27, conv_rate: 0.85

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
