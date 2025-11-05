import time
import logging
from typing import Set

from vllm.v1.request import Request
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.core.sched.utils import remove_all

logger = logging.getLogger(__name__)

class QueueDrafter:
    # Feedback loop factors
    FAILURE_EXP_BACKOFF_FACTOR = 2
    SUCCESS_APPROACH_STEP = 1

    # Need at least 2 tokens for verification to be worth it
    EFFICIENCY_DRAFT_THRESHOLD = 2 

    def __init__(self, waiting: RequestQueue, requests_limit: int):
        self._waiting = waiting # local reference

        self._requests: list[Request] = []
        self._requests_limit = requests_limit

        # Distance
        self._draft_distance_to_waiting_front = 1

        assert(self.FAILURE_EXP_BACKOFF_FACTOR > 1)
        assert(self.SUCCESS_APPROACH_STEP > 0)

    def _allow_request_queue_drafting(self, request: Request) -> bool:
        """
        Determines whether request is qualified to be in the list.
        """
        if request not in self._waiting:
            return False
        return self._waiting.index(request) >= self._draft_distance_to_waiting_front

    def __len__(self) -> int:
        """Get number of requests in queue drafter."""
        return len(self._requests)

    def __contains__(self, request: Request) -> bool:
        """Check whether request is in the queue drafter."""
        return request in self._requests

    def remove_requests(self, requests_to_remove: Set[Request]):
        remove_all(self._requests, requests_to_remove)

    def adjust_draft_distance(self, num_draft_tokens: int):
        # Make adjustments, following exponential backoff
        if num_draft_tokens < self.EFFICIENCY_DRAFT_THRESHOLD:
            self._draft_distance_to_waiting_front = max(
                1, self._draft_distance_to_waiting_front * self.FAILURE_EXP_BACKOFF_FACTOR)
        else:
            self._draft_distance_to_waiting_front = max(
                0, self._draft_distance_to_waiting_front - self.SUCCESS_APPROACH_STEP)
        
        logger.info(f"New drafting factor: {self._draft_distance_to_waiting_front}")

        # Check if any drafting requests should be kicked off the list, given the distance update
        requests_to_remove = set()
        for request in self._requests:
            if not self._allow_request_queue_drafting(request):
                requests_to_remove.add(request)
                logger.info(f"[R {request.request_id}] Kicked off drafting")

        self.remove_requests(requests_to_remove)

    def TODO_update_list_of_requests(self):
        """
        Update drafting list of requests to follow some algorithm.

        TODO: perhaps we want to uniformly compute draft tokens for all queue requests.
        TODO: perhaps we want to kick off requests once they draft some number of tokens.
        """
        pass

    def maybe_add_request(self, request: Request) -> bool:
        """Enable queue drafting for request if there is sufficient time

        Args:
            request (Request): The request to potentially add to the queue drafter.

        Returns:
            bool: True if the request was added to the drafter, False otherwise.
        """
        if len(self._requests) >= self._requests_limit or not self._allow_request_queue_drafting(request):
            logger.info(f"[R {request.request_id}] No drafting. Index: {self._waiting.index(request)}, Factor: {self._draft_distance_to_waiting_front}")
            if len(self._requests) == 0:
                # Decrease distance so drafter doesn't soft-lock at a high distance
                self._draft_distance_to_waiting_front = max(1, self._draft_distance_to_waiting_front - 1)
                logger.info(f"[R {request.request_id}] Decreasing factor to {self._draft_distance_to_waiting_front}")
            
            return False
        
        # Add request to queue drafter
        self._requests.append(request)
        request.queue_drafting_start_time = time.time()
        logger.info(f"[R {request.request_id}] Drafting start time: {request.queue_drafting_start_time}") 
        request.record_event(EngineCoreEventType.QUEUE_DRAFTING_START,
                                request.queue_drafting_start_time)
        return True

    def maybe_remove_request(self, request: Request) -> bool:
        """Remove request from queue drafting if insufficient time remains in queue
        
        Args:
            request (Request): The request to potentially remove from the queue drafter.

        Returns:
            bool: True if the request was removed from the drafter, False otherwise.
        """
        if request not in self._requests:
            return False

        if self._allow_request_queue_drafting(request):
            return False
        
        # Request needs to be removed
        request.queue_drafting_stop_time = time.time()
        logger.info(f"[R {request.request_id}] Drafting stop time: {request.queue_drafting_stop_time}") 

        self._requests.remove(request)
        request.record_event(EngineCoreEventType.QUEUE_DRAFTING_STOP,
                            request.queue_drafting_stop_time)
    
        return True
