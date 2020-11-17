"""API Client for IonQ backends"""
import json

import requests

from .exceptions import *
from .qobj_to_ionq import qobj_to_ionq


class IonQClient:
    """IonQ API Client"""

    def __init__(self, token=None, url=None):
        self._token = token
        self._url = url

    @property
    def _url(self):
        attr_name = f"_{IonQClient.__name__}__url"
        if not hasattr(self, attr_name):
            raise AttributeError(f"{self.__class__.__name__} has no attribute '_url'")
        return self.__url

    @_url.setter
    def _url(self, value):
        if value and value.endswith("/"):
            value = value[:-1]
        self.__url = value

    @property
    def api_headers(self):
        return {
            "Authorization": f"apiKey {self._token}",
            "Content-Type": "application/json",
        }

    def make_path(self, *parts):
        return "/".join([self._url] + list(parts))

    def submit_job(self, job):
        """Submit job to IonQ API
        returns JSON object with status "submitted" and the job's id
        """
        jobAsJSON = qobj_to_ionq(job.qobj)
        req_path = self.make_path("jobs")
        res = requests.post(req_path, data=jobAsJSON, headers=self.api_headers)
        if res.status_code != 200:
            raise IonQAPIError.from_response(res)
        return res.json()

    def retrieve_job(self, job_id):
        """Get job from IonQ API
        returns JSON object with status, and return data if job is complete
        """
        req_path = self.make_path("jobs", job_id)
        res = requests.get(req_path, headers=self.api_headers)
        if res.status_code != 200:
            raise IonQAPIError.from_response(res)
        return res.json()

    def cancel_job(self, job_id):
        """Cancel job that has not yet run
        returns JSON object with status "canceled"
        """
        req_path = self.make_path("jobs", job_id, "status", "cancel")
        res = requests.put(req_path, headers=self.api_headers)
        if res.status_code != 200:
            raise IonQAPIError.from_response(res)
        return res.json()

    def delete_job(self, job_id):
        """Delete a job and associated data from IonQ's servers"""
        req_path = self.make_path("jobs", job_id)
        res = requests.delete(req_path, headers=self.api_headers)
        if res.status_code != 200:
            raise IonQAPIError.from_response(res)
        return res.json()
