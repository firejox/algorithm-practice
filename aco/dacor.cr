require "../nn/fcdnn"
require "./ant"

class DACORNeuralNetwork
  
  @neural_network : FCDNeuralNetwork
  @solutions : Array(Ant)
  @ants : Array(Ant)

  @selected_prob : Float64
  @conv_rate : Float64

  def initialize(**args)
    @random_gen = Random.new
    ant_num = args[:ant_num]
    @neural_network = args[:ann]
    food_num = @neural_network.args_num

    @ants = Array.new(ant_num) { Ant.new(food_num) { 0.0 } }
    @solutions = Array.new(ant_num) { Ant.new(food_num) { @random_gen.rand(-1.0..1.0) } }

    @selected_prob = args[:q]
    @conv_rate = args[:conv_rate]
  end

  def solution_construction(input, output)
    @solutions.each do |ant|
      visited_food = ant.route.each

      @neural_network.update_parameters do
        visited_food.next.as(Float64)
      end

      @neural_network.feed_forward(input)

      ant.length = output.zip(@neural_network.output).sum do |(a, b)|
        (a - b) ** 2
      end
    end

    min_idx = (0...@solutions.size).min_by do |i|
      @solutions[i].length
    end

    @ants.each_with_index do |ant, i|
      ant.route.fill do |j|
        selected = @random_gen.rand > @selected_prob ? i : min_idx

        mu = @solutions[selected].route[j]

        sigma = @conv_rate * @solutions.sum do |sol|
          (sol.route[j] - mu).abs
        end / (@solutions.size - 1)

        @random_gen.normal(mu, sigma)
      end

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
    (0...@solutions.size).each do |i|
      if @ants[i].length < @solutions[i].length
        @solutions[i], @ants[i] = @ants[i], @solutions[i]
      end
    end
  end

  def optimal_route
    @solutions.min_by { |ant| ant.length }.route
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
aco_nn = DACORNeuralNetwork.new ant_num: 10, ann: identity_fn, q: 1e-3, conv_rate: 0.85

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
