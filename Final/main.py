from CNN import CNN

'''
Change to 'Final' directory before run
'''

if __name__ == "__main__":
    train_folder = "miniMNIST" 
    train_label = "miniMNIST/labels.txt"
    
    test_folder = "miniMNIST_test"
    test_label = "miniMNIST_test/labels.txt"
    
    # Initialize conifg
    cnn = CNN("config.txt")

    # Load 
    images, labels = cnn.load_folder(train_folder, train_label)

    # Train
    cnn.train(images, labels, epochs=10, learning_rate=0.1)
    cnn.plot_loss()

    # # Predict
    # accuracy, confusion_matrix = cnn.predict_folder(test_folder, test_label)
    # print(confusion_matrix)
    # print(f"Accuracy: {accuracy:.4f}")
    # cnn.plot_confusion_matrix(confusion_matrix)