import graphviz
import cv2


def visualize_neural_network(
        connectivities,
        skip_connectivity=None):
    dot = graphviz.Digraph()

    neurons_in_layers = [set()]

    for i in range(len(connectivities)):
        neurons_in_layers.append(set())
        for to, from_ in connectivities[i]:
            neurons_in_layers[i].add(from_)
            neurons_in_layers[i+1].add(to)


    for ith_layer, connectivities in enumerate(connectivities):
        for (to, from_) in connectivities[:10]:
            dot.edge(str(ith_layer) + str(from_), str(ith_layer+1) + str(to))

    # Render the graph
    dot.render('neural_network.gv', format='png')

    # Read the image using OpenCV
    image = cv2.imread('neural_network.gv.png')

    # Display the image using OpenCV
    cv2.imshow('Neural Network', image)
    cv2.waitKey(100)


connectivities = ([[2, 17],
  [2, 21],
  [2, 25],
  [5, 33]], [[0, 2],
             [0, 5],
             [1, 2]])
