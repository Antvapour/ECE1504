# Initialize plotting coordinates
epoch_axis  = []
epoch_loss_axis   = []
epoch_error_axis  = []
valid_loss_axis   = []
valid_error_axis  = []

# Start training
with tf.Session() as sess :
    # Restore the session
    saver.restore(sess, path)

    for epoch in range(epoch_num) :
        # train for one epoch
        sess.run(dataset_init_op, feed_dict={X:X_train2_full, Y:y_train2_full,
                    batch_size:train_batch_size})

        for batch in range(train_batch_num) :
            sess.run(optimizer)


        # Compute current loss and error
        sess.run(dataset_init_op, feed_dict={X:X_train2_full, Y:y_train2_full,
                    batch_size:X_train2_full.shape[0]})

        epoch_loss, epoch_acc = sess.run([loss, accuracy])
        epoch_error = (1.0 - epoch_acc) * 100

        print("Epoch =", epoch, "loss =", "{:.9f}".format(epoch_loss),
                "error =", "{:.2f}".format(epoch_error), "per cent")

        # Calculate validation loss and error for this epoch
        sess.run(dataset_init_op, feed_dict={X:X_valid2_full, Y:y_valid2_full,
                    batch_size:X_valid2_full.shape[0]})

        valid_loss, valid_acc = sess.run([loss, accuracy])
        valid_error = (1.0 - valid_acc) * 100

        # Update the coordinates
        epoch_axis.append(epoch)
        epoch_loss_axis.append(epoch_loss)
        epoch_error_axis.append(epoch_error)
        valid_loss_axis.append(valid_loss)
        valid_error_axis.append(valid_error)

        if valid_loss < best_loss :
            save_path = new_saver.save(sess, new_path)
            best_loss = valid_loss
            best_error  = valid_error
            check_count = 0
        else :
            check_count += 1
            if check_count > max_check :
                print("Early stopping!")
                break

    # Calculate test precision using the best model
    new_saver.restore(sess, new_path)
    sess.run(dataset_init_op, feed_dict={X:X_test2, Y:y_test2,
                batch_size:X_test2.shape[0]})
    test_acc, test_correct, test_predict = sess.run([accuracy, correct, predict])
    precision = test_acc * 100
    print("Test precision is {:.2f} per cent".format(precision))

# Show the final validation results
print("Final validation loss =", "{:.9f}".format(best_loss))
print("Final validation error =", "{:.2f}".format(best_error), "per cent")

# Plot and show the graphs
plt.plot(epoch_axis, epoch_loss_axis, label='Training Loss')
plt.plot(epoch_axis, valid_loss_axis, label='Validation Loss')
plt.legend()
plt.show()

plt.plot(epoch_axis, epoch_error_axis, label='Training Error in percentage')
plt.plot(epoch_axis, valid_error_axis, label='Validation Error in percentage')
plt.legend()
plt.show()
