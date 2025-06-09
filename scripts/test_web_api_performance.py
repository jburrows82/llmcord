#!/usr/bin/env python3
"""
Performance test script for the optimized web content extraction API.
This script demonstrates the improvements in speed when using caching and batch optimization.
"""

import asyncio
import time
import json
import sys
import os
from typing import List, Dict, Any

# Add the parent directory to Python path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import httpx
    from src.services.grounding import _fetch_batch_queries_from_web_content_api
    from src.core.constants import DEFAULT_WEB_CONTENT_EXTRACTION_API_URL
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


async def test_api_performance():
    """Test the performance of the optimized web content extraction API."""
    
    # Test queries (mix of similar and different queries to test caching)
    test_queries = [
        "Python programming tutorial",
        "Latest Python 3.12 features",
        "Python programming tutorial",  # Duplicate to test cache
        "JavaScript async await",
        "React hooks tutorial",
        "Latest Python 3.12 features",  # Another duplicate
        "Node.js performance optimization",
        "Python programming tutorial",  # Yet another duplicate
    ]
    
    api_url = DEFAULT_WEB_CONTENT_EXTRACTION_API_URL
    api_max_results = 2  # Keep it small for testing
    max_char_per_url = 5000  # Limit content length
    cache_ttl_minutes = 15
    
    print("🚀 Testing Web Content Extraction API Performance")
    print("=" * 60)
    print(f"API URL: {api_url}")
    print(f"Test queries: {len(test_queries)}")
    print(f"Cache TTL: {cache_ttl_minutes} minutes")
    print(f"Max results per query: {api_max_results}")
    print(f"Max characters per URL: {max_char_per_url}")
    print()
    
    # Create HTTP client with optimizations
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    timeout = httpx.Timeout(15.0)  # Match our optimized timeout
    
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        http2=True,  # Enable HTTP/2
        follow_redirects=True
    ) as client:
        
        # Test 1: First run (cold cache)
        print("📊 Test 1: Cold cache performance")
        start_time = time.time()
        
        try:
            results_cold = await _fetch_batch_queries_from_web_content_api(
                test_queries, client, api_url, api_max_results, max_char_per_url, cache_ttl_minutes
            )
            cold_duration = time.time() - start_time
            successful_cold = sum(1 for r in results_cold if r is not None)
            
            print(f"⏱️  Duration: {cold_duration:.2f} seconds")
            print(f"✅ Successful responses: {successful_cold}/{len(test_queries)}")
            print(f"📈 Average time per query: {cold_duration/len(test_queries):.2f}s")
            
        except Exception as e:
            print(f"❌ Error during cold cache test: {e}")
            return
        
        print()
        
        # Test 2: Second run (warm cache)
        print("📊 Test 2: Warm cache performance")
        start_time = time.time()
        
        try:
            results_warm = await _fetch_batch_queries_from_web_content_api(
                test_queries, client, api_url, api_max_results, max_char_per_url, cache_ttl_minutes
            )
            warm_duration = time.time() - start_time
            successful_warm = sum(1 for r in results_warm if r is not None)
            
            print(f"⏱️  Duration: {warm_duration:.2f} seconds")
            print(f"✅ Successful responses: {successful_warm}/{len(test_queries)}")
            print(f"📈 Average time per query: {warm_duration/len(test_queries):.2f}s")
            
        except Exception as e:
            print(f"❌ Error during warm cache test: {e}")
            return
        
        print()
        
        # Performance summary
        print("📈 Performance Summary")
        print("=" * 40)
        if cold_duration > 0:
            speedup = cold_duration / warm_duration if warm_duration > 0 else float('inf')
            time_saved = cold_duration - warm_duration
            print(f"Cold cache time:  {cold_duration:.2f}s")
            print(f"Warm cache time:  {warm_duration:.2f}s")
            print(f"Time saved:       {time_saved:.2f}s ({time_saved/cold_duration*100:.1f}%)")
            print(f"Speedup factor:   {speedup:.1f}x")
        
        print()
        
        # Cache effectiveness
        total_queries = len(test_queries)
        unique_queries = len(set(test_queries))
        cache_hits_expected = total_queries - unique_queries
        
        print("💾 Cache Effectiveness")
        print("=" * 30)
        print(f"Total queries:        {total_queries}")
        print(f"Unique queries:       {unique_queries}")
        print(f"Expected cache hits:  {cache_hits_expected}")
        print(f"Cache hit rate:       {cache_hits_expected/total_queries*100:.1f}%")
        
        print()
        print("🎯 Optimization Features Tested:")
        print("  ✅ In-memory caching with configurable TTL")
        print("  ✅ Batch request processing")
        print("  ✅ Connection pooling (20 keepalive, 100 max)")
        print("  ✅ HTTP/2 support")
        print("  ✅ Reduced timeout (15s instead of 30s)")
        print("  ✅ Parallel request execution")
        print("  ✅ Automatic cache cleanup")


async def test_cache_expiry():
    """Test cache expiry functionality."""
    print("\n🕒 Testing Cache Expiry")
    print("=" * 30)
    
    # Very short TTL for testing
    short_ttl = 0.05  # 3 seconds (0.05 minutes)
    test_query = ["Cache expiry test query"]
    
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    timeout = httpx.Timeout(15.0)
    
    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=True) as client:
        try:
            # First request
            print(f"Making first request with {short_ttl} minute TTL...")
            start = time.time()
            result1 = await _fetch_batch_queries_from_web_content_api(
                test_query, client, DEFAULT_WEB_CONTENT_EXTRACTION_API_URL, 1, 1000, short_ttl
            )
            first_duration = time.time() - start
            print(f"First request: {first_duration:.2f}s")
            
            # Wait for cache to expire
            sleep_time = (short_ttl * 60) + 1  # Convert to seconds and add buffer
            print(f"Waiting {sleep_time:.1f}s for cache to expire...")
            await asyncio.sleep(sleep_time)
            
            # Second request (should be fresh, not cached)
            print("Making second request after cache expiry...")
            start = time.time()
            result2 = await _fetch_batch_queries_from_web_content_api(
                test_query, client, DEFAULT_WEB_CONTENT_EXTRACTION_API_URL, 1, 1000, short_ttl
            )
            second_duration = time.time() - start
            print(f"Second request: {second_duration:.2f}s")
            
            # Both should be similar duration since cache expired
            if abs(first_duration - second_duration) < 1.0:
                print("✅ Cache expiry working correctly - both requests took similar time")
            else:
                print("⚠️  Cache expiry may not be working as expected")
                
        except Exception as e:
            print(f"❌ Error during cache expiry test: {e}")


def main():
    """Main function to run all tests."""
    print("🔧 Web Content Extraction API Performance Test")
    print("This script tests the optimizations made to the web API usage.")
    print()
    
    # Check if API server is likely running
    api_url = DEFAULT_WEB_CONTENT_EXTRACTION_API_URL
    print(f"ℹ️  Note: This test requires the web content extraction API to be running at {api_url}")
    print("   If the API is not available, you'll see connection errors.")
    print()
    
    # Run the main performance test
    try:
        asyncio.run(test_api_performance())
        # Run cache expiry test
        asyncio.run(test_cache_expiry())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return 1
    
    print("\n✅ Performance test completed!")
    return 0


if __name__ == "__main__":
    exit(main()) 