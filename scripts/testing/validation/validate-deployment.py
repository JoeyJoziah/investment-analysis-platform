#!/usr/bin/env python3
"""
Deployment Validation Script
Ensures all components are properly configured for production
"""

import os
import sys
import json
import yaml
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

class DeploymentValidator:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.successes = []
        
    def log_error(self, message: str):
        self.errors.append(message)
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
    
    def log_warning(self, message: str):
        self.warnings.append(message)
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
    
    def log_success(self, message: str):
        self.successes.append(message)
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
    
    def log_info(self, message: str):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")
    
    def check_file_exists(self, filepath: str, required: bool = True) -> bool:
        """Check if a file exists"""
        path = self.root_dir / filepath
        exists = path.exists()
        
        if not exists and required:
            self.log_error(f"Required file missing: {filepath}")
        elif not exists:
            self.log_warning(f"Optional file missing: {filepath}")
        else:
            self.log_success(f"File exists: {filepath}")
        
        return exists
    
    def validate_docker_files(self):
        """Validate Docker configuration files"""
        self.log_info("Validating Docker files...")
        
        required_files = [
            "Dockerfile.backend",
            "Dockerfile.frontend",
            "docker-compose.production.yml",
            "requirements.production.txt",
        ]
        
        optional_files = [
            "docker-compose.yml",
            "docker-compose.development.yml",
            ".dockerignore",
        ]
        
        for file in required_files:
            self.check_file_exists(file, required=True)
        
        for file in optional_files:
            self.check_file_exists(file, required=False)
    
    def validate_environment_config(self):
        """Validate environment configuration"""
        self.log_info("Validating environment configuration...")
        
        # Check for production environment file
        if not self.check_file_exists(".env.production", required=False):
            if self.check_file_exists(".env.production.example", required=True):
                self.log_warning("Production environment file not found. Copy .env.production.example to .env.production")
        
        # Check for sensitive data in example files
        example_file = self.root_dir / ".env.production.example"
        if example_file.exists():
            with open(example_file, 'r') as f:
                content = f.read()
                
                # Check for placeholder values
                if "your-" not in content.lower() and "change-this" not in content.lower():
                    self.log_warning(".env.production.example might contain real credentials")
    
    def validate_nginx_config(self):
        """Validate Nginx configuration"""
        self.log_info("Validating Nginx configuration...")
        
        nginx_files = [
            "nginx/nginx.conf",
            "nginx/default.conf",
            "nginx/security-headers.conf",
        ]
        
        for file in nginx_files:
            if self.check_file_exists(file, required=True):
                # Validate syntax (basic check)
                filepath = self.root_dir / file
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                    # Check for common issues
                    if "server_tokens off" not in content and "nginx.conf" in file:
                        self.log_warning(f"{file}: server_tokens should be off for security")
                    
                    if "ssl_protocols" in content and "TLSv1" in content and "TLSv1.1" not in content:
                        self.log_warning(f"{file}: Consider disabling old TLS versions")
    
    def validate_scripts(self):
        """Validate deployment scripts"""
        self.log_info("Validating deployment scripts...")
        
        scripts = [
            "deploy.sh",
            "scripts/backup.sh",
            "start-docker.sh",
        ]
        
        for script in scripts:
            if self.check_file_exists(script, required=False):
                # Check if executable
                filepath = self.root_dir / script
                if not os.access(filepath, os.X_OK):
                    self.log_warning(f"{script} is not executable. Run: chmod +x {script}")
    
    def validate_monitoring(self):
        """Validate monitoring configuration"""
        self.log_info("Validating monitoring configuration...")
        
        monitoring_files = [
            "monitoring/prometheus.yml",
        ]
        
        for file in monitoring_files:
            if self.check_file_exists(file, required=False):
                # Validate YAML syntax
                filepath = self.root_dir / file
                try:
                    with open(filepath, 'r') as f:
                        yaml.safe_load(f)
                    self.log_success(f"{file} has valid YAML syntax")
                except yaml.YAMLError as e:
                    self.log_error(f"{file} has invalid YAML: {e}")
    
    def validate_docker_compose(self):
        """Validate Docker Compose configuration"""
        self.log_info("Validating Docker Compose configuration...")
        
        compose_file = self.root_dir / "docker-compose.production.yml"
        if not compose_file.exists():
            self.log_error("docker-compose.production.yml not found")
            return
        
        try:
            # Load and validate compose file
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            # Check for required services
            required_services = [
                "postgres", "redis", "backend", "frontend",
                "celery_worker", "celery_beat"
            ]
            
            services = compose_data.get('services', {})
            for service in required_services:
                if service in services:
                    self.log_success(f"Service configured: {service}")
                    
                    # Check for health checks
                    if 'healthcheck' in services[service]:
                        self.log_success(f"Health check configured for {service}")
                    else:
                        self.log_warning(f"No health check for {service}")
                else:
                    self.log_error(f"Required service missing: {service}")
            
            # Check for volumes
            if 'volumes' in compose_data:
                self.log_success(f"Volumes configured: {', '.join(compose_data['volumes'].keys())}")
            
            # Check for networks
            if 'networks' in compose_data:
                self.log_success(f"Networks configured: {', '.join(compose_data['networks'].keys())}")
            
        except yaml.YAMLError as e:
            self.log_error(f"Invalid Docker Compose YAML: {e}")
        except Exception as e:
            self.log_error(f"Error validating Docker Compose: {e}")
    
    def validate_security(self):
        """Validate security configurations"""
        self.log_info("Validating security configurations...")
        
        # Check for secrets in code
        patterns_to_check = [
            ("*.py", ["password", "secret", "token", "key"]),
            ("*.yml", ["password", "secret", "token"]),
            ("*.yaml", ["password", "secret", "token"]),
        ]
        
        # Check .gitignore
        gitignore = self.root_dir / ".gitignore"
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                gitignore_content = f.read()
                
                critical_patterns = [".env", "*.pem", "*.key", "*.crt"]
                for pattern in critical_patterns:
                    if pattern in gitignore_content:
                        self.log_success(f".gitignore includes: {pattern}")
                    else:
                        self.log_warning(f".gitignore should include: {pattern}")
        else:
            self.log_error(".gitignore file not found")
    
    def validate_dependencies(self):
        """Validate Python dependencies"""
        self.log_info("Validating dependencies...")
        
        req_file = self.root_dir / "requirements.production.txt"
        if not req_file.exists():
            self.log_error("requirements.production.txt not found")
            return
        
        with open(req_file, 'r') as f:
            requirements = f.read()
            
            # Check for important security packages
            security_packages = ["cryptography", "PyJWT", "passlib", "bcrypt"]
            for package in security_packages:
                if package.lower() in requirements.lower():
                    self.log_success(f"Security package present: {package}")
                else:
                    self.log_warning(f"Consider adding security package: {package}")
            
            # Check for monitoring packages
            monitoring_packages = ["prometheus-client", "opentelemetry", "sentry-sdk"]
            for package in monitoring_packages:
                if package.lower() in requirements.lower():
                    self.log_success(f"Monitoring package present: {package}")
    
    def check_docker_installed(self):
        """Check if Docker is installed and running"""
        self.log_info("Checking Docker installation...")
        
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_success(f"Docker installed: {result.stdout.strip()}")
            else:
                self.log_error("Docker not properly installed")
        except FileNotFoundError:
            self.log_error("Docker not found in PATH")
        
        try:
            result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_success(f"Docker Compose installed: {result.stdout.strip()}")
            else:
                self.log_error("Docker Compose not properly installed")
        except:
            self.log_error("Docker Compose not found")
    
    def generate_report(self):
        """Generate validation report"""
        print("\n" + "="*60)
        print("DEPLOYMENT VALIDATION REPORT")
        print("="*60)
        
        print(f"\n{Colors.GREEN}Successes: {len(self.successes)}{Colors.NC}")
        print(f"{Colors.YELLOW}Warnings: {len(self.warnings)}{Colors.NC}")
        print(f"{Colors.RED}Errors: {len(self.errors)}{Colors.NC}")
        
        if self.errors:
            print(f"\n{Colors.RED}ERRORS TO FIX:{Colors.NC}")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}WARNINGS TO REVIEW:{Colors.NC}")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print(f"\n{Colors.GREEN}✓ DEPLOYMENT CONFIGURATION IS VALID{Colors.NC}")
            print("\nYou can proceed with deployment using:")
            print("  ./deploy.sh production")
        else:
            print(f"\n{Colors.RED}✗ DEPLOYMENT CONFIGURATION HAS ERRORS{Colors.NC}")
            print("Please fix the errors before deploying.")
        
        return len(self.errors) == 0
    
    def run(self):
        """Run all validations"""
        print("Starting deployment validation...")
        print("="*60)
        
        self.check_docker_installed()
        self.validate_docker_files()
        self.validate_environment_config()
        self.validate_nginx_config()
        self.validate_scripts()
        self.validate_monitoring()
        self.validate_docker_compose()
        self.validate_security()
        self.validate_dependencies()
        
        return self.generate_report()

if __name__ == "__main__":
    validator = DeploymentValidator()
    success = validator.run()
    sys.exit(0 if success else 1)