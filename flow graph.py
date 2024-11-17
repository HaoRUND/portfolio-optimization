from graphviz import Digraph

# Create a Digraph for the original STEGO training steps
dot_original = Digraph(comment='Original STEGO Training Steps')
dot_original.node('A', 'Input Unlabeled Image Data')
dot_original.node('B', 'Apply Self-Supervised Transformations\n(rotation, scaling, color perturbation, etc.)')
dot_original.node('C', 'Extract Feature Representations Using a Neural Network')
dot_original.node('D', 'Contrastive Learning\n(Calculate Feature Similarity Between Original and Transformed Images)')
dot_original.node('E', 'Feature Matching and Semantic Segmentation')

dot_original.edges(['AB', 'BC', 'CD', 'DE'])

# Create a Digraph for the enhanced STEGO training steps with Adaptive Morphological Segmentation and Median Filtering
dot_enhanced = Digraph(comment='Enhanced STEGO Training Steps')
dot_enhanced.node('A', 'Input Unlabeled Image Data')
dot_enhanced.node('B', 'Apply Self-Supervised Transformations\n(rotation, scaling, color perturbation, etc.)')
dot_enhanced.node('C', 'Extract Feature Representations Using a Neural Network')
dot_enhanced.node('D', 'Contrastive Learning\n(Calculate Feature Similarity Between Original and Transformed Images)')
dot_enhanced.node('E', 'Apply Adaptive Morphological Segmentation\nto Correct Small Classification Errors')
dot_enhanced.node('F', 'Apply Median Filtering\nto Further Reduce Noise and Enhance Smoothness')
dot_enhanced.node('G', 'Feature Matching and Semantic Segmentation')

dot_enhanced.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

# Save and render the graphs
dot_original.render('original_stego_training_steps', format='png', cleanup=True)
dot_enhanced.render('enhanced_stego_training_steps', format='png', cleanup=True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Display the original STEGO training steps image
img_original = mpimg.imread('original_stego_training_steps.png')
plt.figure(figsize=(10, 6))
plt.title('Original STEGO Training Steps')
plt.imshow(img_original)
plt.axis('off')
plt.show()

# Display the enhanced STEGO training steps image
img_enhanced = mpimg.imread('enhanced_stego_training_steps.png')
plt.figure(figsize=(10, 6))
plt.title('Enhanced STEGO Training Steps')
plt.imshow(img_enhanced)
plt.axis('off')
plt.show()
