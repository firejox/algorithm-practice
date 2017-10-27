
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
