import random


class Perceptron:
	def __init__(self):
		# salida = x1*w1 + x2*w2 + bias
		self.w1 = random.uniform(-1, 1)
		self.w2 = random.uniform(-1, 1)
		self.bias = random.uniform(-1, 1)

	def predict(self, x1, x2):
		result = self.w1 * x1 + self.w2 * x2 + self.bias
		return 1 if result >= 0 else 0


def train(perceptron, tablaNAND, learning_rate=0.1, generations=20):
	print("Pesos iniciales:")
	print(
		"w1:", round(perceptron.w1, 2),
		"w2:", round(perceptron.w2, 2),
		"bias:", round(perceptron.bias, 2),
	)

	for generation in range(generations):
		for x1, x2, y_true in tablaNAND:
			y_pred = perceptron.predict(x1, x2)
			error = y_true - y_pred

			perceptron.w1 += learning_rate * error * x1
			perceptron.w2 += learning_rate * error * x2
			perceptron.bias += learning_rate * error

		print(
			f"Generation {generation + 1}: "
			f"w1: {round(perceptron.w1, 2)} "
			f"w2: {round(perceptron.w2, 2)} "
			f"bias: {round(perceptron.bias, 2)}"
		)


def test(perceptron, tablaNAND):
	print("\nResultados (NAND):")
	for x1, x2, _ in tablaNAND:
		result = perceptron.predict(x1, x2)
		print(f"{x1} NAND {x2} = {result}")


def main():
	tablaNAND = [
		(0, 0, 1),
		(0, 1, 1),
		(1, 0, 1),
		(1, 1, 0),
	]

	perceptron = Perceptron()
	train(perceptron, tablaNAND, learning_rate=0.1, generations=20)
	test(perceptron, tablaNAND)


if __name__ == "__main__":
	main()
