--> Overview

This API uses a pre-trained insurance classifier to provide predictions from input customer data. It accepts JSON requests, processes them with a Pydantic schema, and returns insurance class predictions in JSON format. Built using FastAPI for performance, automatic documentation, and easy deployment.

--> Endpoints

POST /predict – accepts customer features and returns predicted insurance category.

--> Deployment

This API can be containerized using the included Dockerfile for easy cloud deployment.