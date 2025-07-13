# Transformet Internals and Attention
- **NLP Fundamentals:** Tokens, Embeddings
- **Transformer Internals:** Attention, Positional Encoding, Multi-Head Attention
- **LLM Core Concepts:** LLM Architectures (GPT, BERT, T5, LLaMA, Mistral, MoEs), Training Paradigm, Recent Transformer Improvements
- **LLM Output Configuration & Prompt Engineering:** Prompting Techniques, Output Parameters


### **Basic Level Questions (1-10)**

**1. Question:** What is the core idea behind the "Attention Is All You Need" paper title, and how does it relate to the Transformer's architecture?

**Answer:** The title "Attention Is All You Need" highlights the revolutionary insight that **recurrent** (RNNs/LSTMs) and **convolutional** (CNNs) layers, which were previously dominant in sequence processing, are not strictly necessary for achieving state-of-the-art results. Instead, the Transformer architecture relies *solely* on the attention mechanism, particularly self-attention, to process sequences effectively and capture long-range dependencies. This allowed for parallel computation, a significant speedup in training.

**2. Question:** At a high level, what are the two main components of the original Transformer architecture?

**Answer:** The original Transformer architecture consists of two main components:

- **An Encoder stack:** Responsible for processing the input sequence and generating a rich, contextualized representation of it.
- **A Decoder stack:** Responsible for generating the output sequence, often based on the encoder's output and previously generated tokens.

**3. Question:** What are the three learned linear projections (matrices) that are used to compute attention in a Transformer, and what do they represent?

**Answer:** The three learned linear projections (weight matrices) are:

- **Query (Q) matrix:** Projects the input embedding into a "query" vector. This represents what the current token is "looking for" in other tokens.
- **Key (K) matrix:** Projects the input embedding into a "key" vector. This represents what each token "offers" for others to find.
- **Value (V) matrix:** Projects the input embedding into a "value" vector. This represents the actual information content that will be "passed" or weighted by the attention scores.

**4. Question:** Explain the purpose of "Positional Encoding" in the Transformer. Why is it needed if self-attention can capture relationships between words?

**Answer:** Positional Encoding is crucial because, unlike RNNs, the Transformer's self-attention mechanism processes all tokens in parallel. This means it inherently has no information about the **order or position** of tokens in a sequence. If you shuffle the words in a sentence, the self-attention mechanism would yield the same results. Positional encoding adds a unique, fixed-size vector to each token's embedding based on its absolute (or relative) position, thereby injecting information about word order into the model. This allows the model to understand the sequence's structure and differentiate between sentences with the same words but different meanings due to order (e.g., "Dog bites man" vs. "Man bites dog").

**5. Question:** What is the formula for Scaled Dot-Product Attention, the fundamental building block of attention in Transformers?

Answer: The formula for Scaled Dot-Product Attention is:

Attention(Q,K,V)=softmax(dkQKT)V

Where:

- Q is the Query matrix.
- K is the Key matrix.
- V is the Value matrix.
- dk is the dimension of the key vectors.
- KT is the transpose of the Key matrix.
- softmax is applied row-wise to get attention weights.

**6. Question:**

Why is the dot product in the attention formula scaled by dk

[](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>) ?

**Answer:**

The scaling by dk

[](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)

 is applied to stabilize the gradients during training. When the dot products of query and key vectors become very large (which can happen with high-dimensional vectors, as dk increases), the softmax function can have extremely small gradients. This pushes the softmax outputs towards hard 0 or 1, making the gradients vanish and hindering the learning process. Dividing by dk

[](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702%0Ac-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14%0Ac0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54%0Ac44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10%0As173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429%0Ac69,-144,104.5,-217.7,106.5,-221%0Al0 -0%0Ac5.3,-9.3,12,-14,20,-14%0AH400000v40H845.2724%0As-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7%0Ac-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z%0AM834 80h400000v40h-400000z"></path></svg>)

 helps to keep the variance of the dot products consistent, preventing the softmax from saturating and ensuring more stable training.

**7. Question:** What is "Multi-Head Attention," and why is it used instead of a single attention head?

**Answer:** "Multi-Head Attention" is a mechanism that allows the Transformer model to jointly attend to information from different "representation subspaces" at different positions. Instead of performing a single attention1 calculation, it performs multiple attention calculations in parallel, each with its own independent learned Query, Key, and Value projection matrices. The outputs from these multiple "heads" are then concatenated and linearly transformed.

It's used because:

- **Diverse Perspectives:** Each head can learn to focus on different aspects of the input sequence or different types of relationships (e.g., one head might focus on syntactic relationships, another on semantic relationships).
- **Richer Representation:** By combining these diverse perspectives, the model gains a more comprehensive and robust understanding of the input, improving its ability to capture complex dependencies.

**8. Question:** Describe the basic structure of a single "Encoder Layer" in the Transformer architecture.

**Answer:** A single Encoder Layer in the Transformer typically consists of two main sub-layers, each followed by a Residual Connection and Layer Normalization:

1. **Multi-Head Self-Attention:** Processes the input sequence by allowing each token to attend to all other tokens in the *same* sequence.
2. **Position-wise Feed-Forward Network (FFN):** A simple fully connected neural network applied independently and identically to each position (token) in the sequence. It typically has two linear transformations with a ReLU activation in between.

**9. Question:** What is the purpose of the "Residual Connections" (or skip connections) in the Transformer, and where are they applied?

**Answer:** Residual Connections (or skip connections) involve adding the input of a sub-layer to its output before Layer Normalization. They are applied around both the Multi-Head Attention sub-layer and the Feed-Forward Network sub-layer within each Transformer block (both encoder and decoder). Their purpose is:

- **Mitigate Vanishing Gradients:** Helps gradients flow more directly through the network, allowing for the training of very deep models.
- **Easier Optimization:** Makes it easier for the network to learn identity mappings, meaning if a layer doesn't need to change the input, it can simply pass it through.
- **Improved Performance:** Generally leads to better performance and faster convergence.

**10. Question:** What is "Layer Normalization," and why is it used in the Transformer (as opposed to Batch Normalization)?

Answer: "Layer Normalization" normalizes the inputs across the features (embedding dimensions) for each individual sample in a batch, independently of other samples. For a given token's embedding vector, it normalizes its values across its dimensions.

It's used in Transformers (often preferred over Batch Normalization) because:

- **Sequence Length Variability:** Batch Normalization's effectiveness can be hampered by variable sequence lengths in NLP tasks. Layer Normalization operates independently per sequence, making it more robust to varying sequence lengths.
- **Small Batch Sizes:** For very large models, batch sizes can be small due to memory constraints. Batch Normalization performs poorly with small batch sizes, whereas Layer Normalization is unaffected.
- **Stability:** It helps stabilize training, especially in deep networks, by keeping the input distribution to activation functions consistent.
