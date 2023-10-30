obs_id = np.arange(n)  # [0, 1, ..., n-1]
# Run the whole data set `nepoch` times
for i in range(nepoch):
    # Shuffle observation IDs
    np.random.shuffle(obs_id)

    # Update on mini-batches
    for j in range(0, n, batch_size):
        # Create mini-batch
        x_mini_batch = x[obs_id[j:(j + batch_size)]]
        # Compute loss
        loss = model(x_mini_batch)
        # Compute gradient and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
