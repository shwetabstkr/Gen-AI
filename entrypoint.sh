#!/bin/bash
set -e
ollama serve &
sleep 5
ollama pull llama3
wait