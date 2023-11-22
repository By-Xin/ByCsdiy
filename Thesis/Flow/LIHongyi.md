# LIHongyi - VAE & Generative Model
https://openai.com/blog/generative

## PixelRNN
- Image generation
    ![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311161102445.png)

- Audio Processing

## Variational Autoencoder (VAE)

### Pre VAE: Auto-encoder

***For training:***

```mermaid
graph LR
    A[Input picture] --> B(NN Encoder)
    B --> C[CODE]
    C --> D(NN Decoder)
    D --> E[Output picture]
```

And here we want the output picture to be as similar as the input picture.

***For generation:***

```mermaid
graph LR
    A[Random Vector Code] --> B(NN Decoder)
    B --> C[Output picture]
```

### VAE

![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311161125356.png)

- The goal is to :
  - Minimize reconstruction error (difference between input and output)
  - Minimize $\sum_{i=1}^3(1+\sigma_i-m^2-\exp(\sigma_i))$