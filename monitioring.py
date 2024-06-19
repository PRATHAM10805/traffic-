import matplotlib.pyplot as plt
from modeldevelopment import y_pred,y_test
# Plot actual vs predicted traffic volume
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.legend()
plt.show()
