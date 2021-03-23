Installing the Qiskit IonQ Provider
===================================

You can install the provider using pip::

   pip install qiskit-ionq

Provider Setup
--------------

The IonQ Provider uses IonQ's REST API.

To instantiate the provider, make sure you have an access token then create a provider::


   from qiskit_ionq import IonQProvider

   provider = IonQProvider("superseekr!t-token")


Credential Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, the IonQ Provider can discover your access token using the ``QISKIT_IONQ_API_TOKEN`` environment variable::

   export QISKIT_IONQ_API_TOKEN="superseekr!t-token"

Then instantiate the provider without any arguments::

   from qiskit_ionq import IonQProvider

   provider = IonQProvider()
