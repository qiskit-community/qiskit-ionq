Installing the Qiskit IonQ Provider
===================================

.. NOTE::

      **The Qiskit IonQ Provider requires** ``qiskit-terra>=0.17.4``!

      To ensure you are on the latest version, run::

         pip install -U "qiskit-terra>=0.17.4"


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

Alternatively, the IonQ Provider can discover your access token from an environment variable. It checks ``QISKIT_IONQ_API_TOKEN``, then ``IONQ_API_KEY``, then ``IONQ_API_TOKEN``::

   export IONQ_API_KEY="token"

Then instantiate the provider without any arguments::

   from qiskit_ionq import IonQProvider

   provider = IonQProvider()
