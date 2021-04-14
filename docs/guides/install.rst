Installing the Qiskit IonQ Provider
===================================

.. WARNING::

      **The Qiskit IonQ Provider requires** ``qiskit<=0.24.0``!

      To ensure you are on the latest version, run::

         pip install -U "qiskit>=0.25.0"


You can install the provider using pip::

   pip install qiskit-ionq

Provider Setup
--------------

The IonQ Provider uses IonQ's REST API.

To instantiate the provider, make sure you have an access token then create a provider::


   from qiskit_ionq import IonQProvider

   provider = IonQProvider("token")


Credential Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, the IonQ Provider can discover your access token using the ``QISKIT_IONQ_API_TOKEN`` environment variable::

   export QISKIT_IONQ_API_TOKEN="token"

Then instantiate the provider without any arguments::

   from qiskit_ionq import IonQProvider

   provider = IonQProvider()
