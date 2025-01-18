# Sample Markdown File for Vector DB Testing

## Introduction
This document provides an example Markdown file designed to test ingestion and querying of Markdown files into a vector DB. The file includes nonsensical questions and answers to simulate real-world data scenarios.

## Document Structure
The following sections demonstrate various Markdown formatting capabilities:

### Headers and Subheaders

# Main Heading
## Secondary Heading
#### Tertiary Heading

1. Ordered List Item 1
2. Ordered List Item 2
3. Ordered List Item 3

- Unordered List Item 1
- Unordered List Item 2
- Unordered List Item 3

### Images and Links

![Example Image](https://example.com/image.jpg)
[Link Text](https://example.com/link)

### Code Blocks and Syntax Highlighting

```markdown
This is a code block with syntax highlighting.
```

```python
print("Hello, World!")
```

### Tables

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2    |

### Footnotes and Citations

[^1] This is a footnote.

[^2] [Example Citation](https://example.com/citation)

## Testing Data
The following sections contain nonsensical questions and answers to test the effectiveness of embeddings and vector queries:

### Question 1: What is the airspeed velocity of an unladen swallow?

Answer: 11 meters per second.

### Question 2: Can cats fit inside a tea cup?

Answer: No, but they can fit inside a paperclip.

### Question 3: How many rainbows can fit in a teaspoon?

Answer: None, because rainbows are color spectrums that don't have a physical presence.

## Embedding and Query Testing
To test the effectiveness of embeddings and vector queries, we can use the following query format:

`vec_query { embedding_vector } @vector_db`

Example query:
```markdown
vec_query {"color": "blue", "shape": "circle"} @vector_db
```
This query searches for documents with an embedding vector that matches the specified colors (blue) and shapes (circle).

Note: This is just a starting point, and you can modify or add sections as needed to suit your testing requirements.