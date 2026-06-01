
import boto3
import json
import os

ENDPOINT_NAME = "credit-risk-realtime"
sm_runtime = boto3.client("sagemaker-runtime")

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        features = body.get("features", [])

        if not features:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "features list is required"})
            }

        payload = ",".join(str(v) for v in features)

        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload
        )

        score = float(response["Body"].read().decode().strip())
        risk  = "HIGH" if score >= 0.5 else "LOW"

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "default_probability": round(score, 4),
                "risk_label":          risk,
                "threshold":           0.5,
                "model_endpoint":      ENDPOINT_NAME
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
