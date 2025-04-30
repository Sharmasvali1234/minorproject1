"""
Gunicorn configuration for optimized performance on Render
"""

# Increase timeout to prevent worker termination during video processing
timeout = 300  # 5 minutes

# Use 1 worker to minimize memory consumption
workers = 1

# Use sync worker class for reliability
worker_class = 'sync'

# Bind to the port Render expects
bind = '0.0.0.0:10000'

# Reduce logging verbosity
loglevel = 'info'

# Disable request line logging for better performance
access_log_format = '%(r)s %(s)s %(b)s %(L)s'

# Set maximum request body size
limit_request_line = 0
limit_request_fields = a100
limit_request_field_size = 0

# Configure worker process handling
graceful_timeout = 120
keepalive = 5