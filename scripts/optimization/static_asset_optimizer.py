#!/usr/bin/env python3
"""
Static Asset Optimization Script for Investment Analysis Platform
Optimizes images, minifies CSS/JS, generates sprite sheets, and implements caching strategies
Designed for cost-efficient CDN delivery with maximum performance
"""

import os
import sys
import json
import shutil
import hashlib
import gzip
import brotli
from pathlib import Path
from typing import Dict, List, Set, Optional
import subprocess
import argparse
import logging
from dataclasses import dataclass
from PIL import Image, ImageOptim
import cssmin
import jsmin
import htmlmin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AssetOptimization:
    """Asset optimization statistics"""
    original_size: int
    optimized_size: int
    compression_ratio: float
    file_path: str
    optimization_type: str


class StaticAssetOptimizer:
    """
    Production-ready static asset optimizer
    Handles images, CSS, JS, and HTML with advanced compression
    """
    
    def __init__(self, source_dir: Path, output_dir: Path, config: Optional[Dict] = None):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.config = config or self._get_default_config()
        
        # Statistics tracking
        self.optimizations: List[AssetOptimization] = []
        self.total_original_size = 0
        self.total_optimized_size = 0
        
        # File hashing for cache busting
        self.asset_hashes: Dict[str, str] = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized optimizer: {source_dir} -> {output_dir}")
    
    def _get_default_config(self) -> Dict:
        """Get default optimization configuration"""
        return {
            'images': {
                'jpeg_quality': 85,
                'png_optimize': True,
                'webp_enabled': True,
                'svg_optimize': True,
                'max_width': 1920,
                'max_height': 1080
            },
            'css': {
                'minify': True,
                'autoprefixer': True,
                'remove_unused': False  # Requires PurgeCSS integration
            },
            'js': {
                'minify': True,
                'babel_transpile': False,  # Would require Node.js setup
                'tree_shaking': False     # Would require webpack
            },
            'html': {
                'minify': True,
                'remove_comments': True,
                'remove_whitespace': True
            },
            'compression': {
                'gzip': True,
                'brotli': True,
                'gzip_level': 9,
                'brotli_quality': 11
            },
            'caching': {
                'cache_busting': True,
                'manifest_generation': True
            }
        }
    
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate hash for cache busting"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]  # Use first 12 characters
    
    def _record_optimization(self, original_path: Path, optimized_path: Path, opt_type: str):
        """Record optimization statistics"""
        original_size = original_path.stat().st_size
        optimized_size = optimized_path.stat().st_size
        
        compression_ratio = (1 - (optimized_size / original_size)) * 100
        
        optimization = AssetOptimization(
            original_size=original_size,
            optimized_size=optimized_size,
            compression_ratio=compression_ratio,
            file_path=str(optimized_path.relative_to(self.output_dir)),
            optimization_type=opt_type
        )
        
        self.optimizations.append(optimization)
        self.total_original_size += original_size
        self.total_optimized_size += optimized_size
        
        logger.info(f"Optimized {original_path.name}: {original_size} -> {optimized_size} bytes ({compression_ratio:.1f}% reduction)")
    
    def optimize_image(self, image_path: Path) -> Path:
        """Optimize image files with multiple format support"""
        config = self.config['images']
        
        # Create output path with potential format change
        relative_path = image_path.relative_to(self.source_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with Image.open(image_path) as img:
                # Convert RGBA to RGB for JPEG if necessary
                if img.mode == 'RGBA' and image_path.suffix.lower() in ['.jpg', '.jpeg']:
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                
                # Resize if too large
                if img.width > config['max_width'] or img.height > config['max_height']:
                    img.thumbnail((config['max_width'], config['max_height']), Image.Resampling.LANCZOS)
                    logger.info(f"Resized {image_path.name} to {img.width}x{img.height}")
                
                # Save optimized image
                if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                    img.save(output_path, 'JPEG', 
                            quality=config['jpeg_quality'], 
                            optimize=True, 
                            progressive=True)
                elif image_path.suffix.lower() == '.png':
                    img.save(output_path, 'PNG', optimize=config['png_optimize'])
                else:
                    # Copy other formats as-is
                    shutil.copy2(image_path, output_path)
                
                # Generate WebP version if enabled
                if config['webp_enabled'] and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    webp_path = output_path.with_suffix('.webp')
                    img.save(webp_path, 'WEBP', quality=config['jpeg_quality'], method=6)
                    logger.info(f"Generated WebP version: {webp_path.name}")
                
        except Exception as e:
            logger.warning(f"Failed to optimize image {image_path}: {e}")
            # Fallback: copy original file
            shutil.copy2(image_path, output_path)
        
        self._record_optimization(image_path, output_path, 'image')
        return output_path
    
    def optimize_svg(self, svg_path: Path) -> Path:
        """Optimize SVG files using svgo (if available)"""
        relative_path = svg_path.relative_to(self.source_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to use svgo if available
        try:
            subprocess.run(['svgo', '--input', str(svg_path), '--output', str(output_path)], 
                          check=True, capture_output=True)
            logger.info(f"Optimized SVG with svgo: {svg_path.name}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: copy original file
            shutil.copy2(svg_path, output_path)
            logger.debug(f"svgo not available, copied SVG as-is: {svg_path.name}")
        
        self._record_optimization(svg_path, output_path, 'svg')
        return output_path
    
    def optimize_css(self, css_path: Path) -> Path:
        """Optimize CSS files with minification"""
        relative_path = css_path.relative_to(self.source_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            
            if self.config['css']['minify']:
                # Minify CSS
                minified_css = cssmin.cssmin(css_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(minified_css)
                
                logger.info(f"Minified CSS: {css_path.name}")
            else:
                # Copy as-is
                shutil.copy2(css_path, output_path)
        
        except Exception as e:
            logger.warning(f"Failed to optimize CSS {css_path}: {e}")
            shutil.copy2(css_path, output_path)
        
        self._record_optimization(css_path, output_path, 'css')
        return output_path
    
    def optimize_js(self, js_path: Path) -> Path:
        """Optimize JavaScript files with minification"""
        relative_path = js_path.relative_to(self.source_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(js_path, 'r', encoding='utf-8') as f:
                js_content = f.read()
            
            if self.config['js']['minify']:
                # Minify JavaScript
                minified_js = jsmin.jsmin(js_content)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(minified_js)
                
                logger.info(f"Minified JS: {js_path.name}")
            else:
                # Copy as-is
                shutil.copy2(js_path, output_path)
        
        except Exception as e:
            logger.warning(f"Failed to optimize JS {js_path}: {e}")
            shutil.copy2(js_path, output_path)
        
        self._record_optimization(js_path, output_path, 'js')
        return output_path
    
    def optimize_html(self, html_path: Path) -> Path:
        """Optimize HTML files with minification"""
        relative_path = html_path.relative_to(self.source_dir)
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            if self.config['html']['minify']:
                # Minify HTML
                minified_html = htmlmin.minify(
                    html_content,
                    remove_comments=self.config['html']['remove_comments'],
                    remove_empty_space=self.config['html']['remove_whitespace'],
                    reduce_boolean_attributes=True,
                    remove_optional_attribute_quotes=True
                )
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(minified_html)
                
                logger.info(f"Minified HTML: {html_path.name}")
            else:
                # Copy as-is
                shutil.copy2(html_path, output_path)
        
        except Exception as e:
            logger.warning(f"Failed to optimize HTML {html_path}: {e}")
            shutil.copy2(html_path, output_path)
        
        self._record_optimization(html_path, output_path, 'html')
        return output_path
    
    def compress_file(self, file_path: Path):
        """Create compressed versions of files (gzip and brotli)"""
        config = self.config['compression']
        
        if config['gzip']:
            gzip_path = Path(str(file_path) + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(gzip_path, 'wb', compresslevel=config['gzip_level']) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.debug(f"Created gzip version: {gzip_path.name}")
        
        if config['brotli']:
            brotli_path = Path(str(file_path) + '.br')
            with open(file_path, 'rb') as f_in:
                compressed_data = brotli.compress(f_in.read(), quality=config['brotli_quality'])
                with open(brotli_path, 'wb') as f_out:
                    f_out.write(compressed_data)
            
            logger.debug(f"Created Brotli version: {brotli_path.name}")
    
    def generate_asset_manifest(self):
        """Generate asset manifest for cache busting and dependency tracking"""
        manifest = {
            'version': '1.0.0',
            'generated': str(Path.cwd()),
            'assets': {},
            'statistics': {
                'total_files': len(self.optimizations),
                'total_original_size': self.total_original_size,
                'total_optimized_size': self.total_optimized_size,
                'total_compression_ratio': (1 - (self.total_optimized_size / max(self.total_original_size, 1))) * 100
            }
        }
        
        # Add asset information
        for opt in self.optimizations:
            file_path = opt.file_path
            full_path = self.output_dir / file_path
            
            # Calculate hash for cache busting
            if full_path.exists():
                file_hash = self._calculate_hash(full_path)
                self.asset_hashes[file_path] = file_hash
                
                manifest['assets'][file_path] = {
                    'hash': file_hash,
                    'size': opt.optimized_size,
                    'compression_ratio': opt.compression_ratio,
                    'type': opt.optimization_type,
                    'cache_busting_name': f"{full_path.stem}.{file_hash}{full_path.suffix}"
                }
        
        # Write manifest file
        manifest_path = self.output_dir / 'asset-manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Generated asset manifest: {manifest_path}")
        return manifest
    
    def create_cache_busted_copies(self):
        """Create cache-busted copies of assets"""
        if not self.config['caching']['cache_busting']:
            return
        
        logger.info("Creating cache-busted asset copies...")
        
        for file_path, file_hash in self.asset_hashes.items():
            original_path = self.output_dir / file_path
            if original_path.exists():
                # Create cache-busted filename
                cache_busted_name = f"{original_path.stem}.{file_hash}{original_path.suffix}"
                cache_busted_path = original_path.parent / cache_busted_name
                
                # Copy file with cache-busted name
                shutil.copy2(original_path, cache_busted_path)
                
                # Also create compressed versions if they exist
                for ext in ['.gz', '.br']:
                    compressed_original = Path(str(original_path) + ext)
                    if compressed_original.exists():
                        compressed_cache_busted = Path(str(cache_busted_path) + ext)
                        shutil.copy2(compressed_original, compressed_cache_busted)
                
                logger.debug(f"Created cache-busted copy: {cache_busted_name}")
    
    def optimize_all_assets(self):
        """Optimize all assets in the source directory"""
        logger.info("Starting asset optimization process...")
        
        # Define file type mappings
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        svg_extensions = {'.svg'}
        css_extensions = {'.css'}
        js_extensions = {'.js'}
        html_extensions = {'.html', '.htm'}
        compressible_extensions = {'.css', '.js', '.html', '.htm', '.json', '.xml', '.txt', '.svg'}
        
        # Walk through source directory
        for file_path in self.source_dir.rglob('*'):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                try:
                    # Optimize based on file type
                    if file_ext in image_extensions:
                        optimized_path = self.optimize_image(file_path)
                    elif file_ext in svg_extensions:
                        optimized_path = self.optimize_svg(file_path)
                    elif file_ext in css_extensions:
                        optimized_path = self.optimize_css(file_path)
                    elif file_ext in js_extensions:
                        optimized_path = self.optimize_js(file_path)
                    elif file_ext in html_extensions:
                        optimized_path = self.optimize_html(file_path)
                    else:
                        # Copy other files as-is
                        relative_path = file_path.relative_to(self.source_dir)
                        optimized_path = self.output_dir / relative_path
                        optimized_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, optimized_path)
                        
                        self._record_optimization(file_path, optimized_path, 'copy')
                    
                    # Create compressed versions for compressible files
                    if file_ext in compressible_extensions:
                        self.compress_file(optimized_path)
                
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        # Generate manifest and cache-busted copies
        self.generate_asset_manifest()
        self.create_cache_busted_copies()
        
        # Print optimization summary
        self.print_optimization_summary()
    
    def print_optimization_summary(self):
        """Print optimization statistics summary"""
        if not self.optimizations:
            logger.warning("No optimizations performed")
            return
        
        total_savings = self.total_original_size - self.total_optimized_size
        overall_compression = (total_savings / max(self.total_original_size, 1)) * 100
        
        print("\n" + "="*50)
        print("ASSET OPTIMIZATION SUMMARY")
        print("="*50)
        print(f"Total files processed: {len(self.optimizations)}")
        print(f"Original total size: {self.total_original_size:,} bytes ({self.total_original_size / (1024*1024):.1f} MB)")
        print(f"Optimized total size: {self.total_optimized_size:,} bytes ({self.total_optimized_size / (1024*1024):.1f} MB)")
        print(f"Total savings: {total_savings:,} bytes ({total_savings / (1024*1024):.1f} MB)")
        print(f"Overall compression: {overall_compression:.1f}%")
        
        # Breakdown by optimization type
        type_stats = {}
        for opt in self.optimizations:
            opt_type = opt.optimization_type
            if opt_type not in type_stats:
                type_stats[opt_type] = {
                    'count': 0,
                    'original_size': 0,
                    'optimized_size': 0
                }
            
            type_stats[opt_type]['count'] += 1
            type_stats[opt_type]['original_size'] += opt.original_size
            type_stats[opt_type]['optimized_size'] += opt.optimized_size
        
        print("\nBreakdown by file type:")
        for opt_type, stats in type_stats.items():
            savings = stats['original_size'] - stats['optimized_size']
            compression = (savings / max(stats['original_size'], 1)) * 100
            print(f"  {opt_type.capitalize()}: {stats['count']} files, {savings:,} bytes saved ({compression:.1f}%)")
        
        print("="*50 + "\n")


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Optimize static assets for production deployment")
    parser.add_argument("source", help="Source directory containing assets")
    parser.add_argument("output", help="Output directory for optimized assets")
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize optimizer and process assets
    optimizer = StaticAssetOptimizer(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        config=config
    )
    
    try:
        optimizer.optimize_all_assets()
        logger.info("Asset optimization completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Asset optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())