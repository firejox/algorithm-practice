require "../nn/fcdnn"
require "./ant"

class ACORNeuralNetwork
  @solution_size : Int32
  @ants : Array(Ant)
  @neural_network : FCDNeuralNetwork
  
  @locality : Float64
  @coeff_prob : Float64
  @conv_rate : Float64

  def initialize(**args)
    @random_gen = Random.new
    ant_num = args[:ant_num]
    @neural_network = args[:ann]
    food_num = @neural_network.args_num
    
    @solution_size = Math.max(food_num, 10)

    @ants = Array.new(ant_num + @solution_size) do |i|
      if i < @solution_size
        Ant.new(food_num) { @random_gen.rand(-1.0..1.0) }
      else
        Ant.new(food_num) { 0.0 }
      end
    end

    @locality = args[:locality]
    @conv_rate = args[:conv_rate]

    @coeff_prob = 1.0 - @locality ** @solution_size
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

    (ant_slice + @solution_size).each do |ant|
      ant.route.fill do |i|
        rnd = @random_gen.rand
        selected = Math.max(Math.log(1.0 - rnd * @coeff_prob, @locality).to_i - 1, 0)
        mu = @ants[selected].route[i]
        
        sigma = @conv_rate * ant_slice[0, @solution_size - 1].sum do |ant|
          (ant.route[i] - mu).abs
        end / (@solution_size - 1)

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
aco_nn = ACORNeuralNetwork.new ant_num: 2, ann: identity_fn, locality: 0.1, conv_rate: 0.85

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
