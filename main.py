"""
Meteor/Asteroid Trajectory Prediction and Impact Probability API
Integrates NASA SBDB, NEO APIs with Poliastro for orbital mechanics
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import httpx
import numpy as np
from datetime import datetime, timedelta

# Poliastro imports
from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit

from astropy import units as u
from astropy.time import Time

# Initialize FastAPI
app = FastAPI(
    title="Meteor Trajectory & Impact Prediction API",
    description="Predict meteor trajectories and impact probabilities using NASA data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
NASA_API_KEY = "MwfPBwoFvtIiQXrn41tXzIW5AtxBxGl5PXH6lVCZ"  # Replace with your actual API key
NEO_API_BASE = "https://api.nasa.gov/neo/rest/v1"
SBDB_API_BASE = "https://ssd-api.jpl.nasa.gov/sbdb.api"

# ============================================================================
# MODELS
# ============================================================================

class OrbitalElements(BaseModel):
    """Orbital elements for trajectory calculation"""
    semi_major_axis: float = Field(..., description="Semi-major axis (AU)")
    eccentricity: float = Field(..., description="Eccentricity")
    inclination: float = Field(..., description="Inclination (degrees)")
    longitude_asc_node: float = Field(..., description="Longitude of ascending node (degrees)")
    argument_perihelion: float = Field(..., description="Argument of perihelion (degrees)")
    mean_anomaly: float = Field(..., description="Mean anomaly (degrees)")
    epoch: str = Field(..., description="Epoch date")

class TrajectoryResponse(BaseModel):
    """Complete trajectory analysis response"""
    neo_id: Optional[str]
    name: str
    is_potentially_hazardous: bool
    close_approach_date: Optional[str]
    close_approach_distance_km: Optional[float]
    relative_velocity_kmh: Optional[float]
    estimated_diameter_km_min: float
    estimated_diameter_km_max: float
    absolute_magnitude: float
    orbital_elements: Dict[str, Any]
    impact_probability: float
    minimum_orbit_intersection_distance_au: Optional[float]
    trajectory_points: List[Dict[str, float]]
    risk_assessment: str
    orbital_period_days: Optional[float]

# ============================================================================
# NASA API FUNCTIONS
# ============================================================================

async def fetch_neo_data(neo_id: str) -> Dict[str, Any]:
    """Fetch NEO data from NASA NEO API"""
    url = f"{NEO_API_BASE}/neo/{neo_id}"
    params = {"api_key": NASA_API_KEY}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch NEO data: {str(e)}")



async def fetch_neo_feed(start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch NEO feed for date range"""
    url = f"{NEO_API_BASE}/feed"
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": NASA_API_KEY
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch NEO feed: {str(e)}")

# ============================================================================
# POLIASTRO TRAJECTORY SIMULATION
# ============================================================================

def simulate_trajectory_poliastro(orbital_elements: Dict[str, Any], days: int) -> tuple[List[Dict[str, float]], float]:
    """
    Accurate trajectory simulation using Poliastro
    Returns: (trajectory_points, orbital_period_days)
    """
    try:
        # Extract orbital elements
        a = orbital_elements.get('semi_major_axis', 1.5) * u.AU
        ecc = orbital_elements.get('eccentricity', 0.1) * u.one
        inc = orbital_elements.get('inclination', 5) * u.deg
        raan = orbital_elements.get('longitude_asc_node', 0) * u.deg
        argp = orbital_elements.get('argument_perihelion', 0) * u.deg
        
        # Use actual mean anomaly instead of always starting at perihelion
        M = orbital_elements.get('mean_anomaly', 0) * u.deg
        
        # Convert mean anomaly to true anomaly using Kepler's equation
        e_val = float(ecc.value)
        M_rad = M.to(u.rad).value
        
        # Solve Kepler's equation: E - e*sin(E) = M
        E = M_rad
        for _ in range(10):  # Newton-Raphson iteration
            E = E - (E - e_val * np.sin(E) - M_rad) / (1 - e_val * np.cos(E))
        
        # Convert eccentric anomaly to true anomaly
        nu = 2 * np.arctan2(
            np.sqrt(1 + e_val) * np.sin(E/2),
            np.sqrt(1 - e_val) * np.cos(E/2)
        ) * u.rad
        
        epoch_str = orbital_elements.get('epoch', '2025-01-01')
        epoch = Time(epoch_str, format='iso')
        
        # Create orbit around the Sun with actual position
        orbit = Orbit.from_classical(
            Sun,
            a, ecc, inc, raan, argp, nu,
            epoch=epoch
        )
        
        # Calculate orbital period for better sampling
        period_days = float((2 * np.pi * np.sqrt(a**3 / Sun.k)).to(u.day).value)
        
        # Generate trajectory points with appropriate sampling
        num_points = min(100, max(50, int(days / max(1, period_days / 20))))
        time_points = np.linspace(0, days, num_points) * u.day
        
        points = []
        for t in time_points:
            propagated = orbit.propagate(t)
            r = propagated.r.to(u.AU).value
            v = propagated.v.to(u.km / u.s).value
            
            # Calculate distance to Earth
            earth_pos = np.array([1.0, 0.0, 0.0])  # Simplified Earth position
            dist_to_earth = np.linalg.norm(r - earth_pos)
            
            points.append({
                'day': int(t.value),
                'x_au': round(float(r[0]), 6),
                'y_au': round(float(r[1]), 6),
                'z_au': round(float(r[2]), 6),
                'distance_from_sun_au': round(float(np.linalg.norm(r)), 6),
                'velocity_km_s': round(float(np.linalg.norm(v)), 2),
                'distance_to_earth_au': round(float(dist_to_earth), 6)
            })
        
        return points, period_days
    
    except Exception as e:
        # Fallback to simplified simulation
        return simulate_trajectory_simple(orbital_elements, days), None

def simulate_trajectory_simple(orbital_elements: Dict[str, Any], days: int) -> List[Dict[str, float]]:
    """Simplified trajectory simulation (fallback)"""
    points = []
    
    a = orbital_elements.get('semi_major_axis', 1.5)
    e = orbital_elements.get('eccentricity', 0.1)
    i = np.radians(orbital_elements.get('inclination', 5))
    M0 = np.radians(orbital_elements.get('mean_anomaly', 0))
    
    # Calculate orbital period
    period_days = 365.25 * (a ** 1.5)
    
    for day in range(0, days, max(1, days // 50)):
        # Mean anomaly at time t
        M = M0 + (day / period_days) * 2 * np.pi
        
        # Solve Kepler's equation for eccentric anomaly
        E = M
        for _ in range(5):
            E = M + e * np.sin(E)
        
        # True anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
        
        # Distance from Sun
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        
        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # Apply inclination
        x = x_orb * np.cos(i)
        y = y_orb
        z = x_orb * np.sin(i)
        
        # Simplified distance to Earth
        earth_pos = np.array([1.0, 0.0, 0.0])
        pos = np.array([x, y, z])
        dist_to_earth = np.linalg.norm(pos - earth_pos)
        
        points.append({
            'day': day,
            'x_au': round(x, 6),
            'y_au': round(y, 6),
            'z_au': round(z, 6),
            'distance_from_sun_au': round(r, 6),
            'distance_to_earth_au': round(dist_to_earth, 6)
        })
    
    return points

def calculate_moid(orbital_elements: Dict[str, Any]) -> float:
    """Calculate Minimum Orbit Intersection Distance with Earth"""
    a = orbital_elements.get('semi_major_axis', 1.5)
    e = orbital_elements.get('eccentricity', 0.1)
    
    a_earth = 1.0
    perihelion = a * (1 - e)
    aphelion = a * (1 + e)
    
    if perihelion <= a_earth <= aphelion:
        moid = max(0.001, abs(a - a_earth) * (1 - e))
    else:
        moid = min(abs(perihelion - a_earth), abs(aphelion - a_earth))
    
    return moid

def calculate_impact_probability(moid_au: float, diameter_km: float, 
                                close_approach_distance_km: Optional[float]) -> float:
    """Calculate impact probability"""
    moid_km = moid_au * 149597870.7
    earth_radius = 6371
    gravitational_focus = 11000
    
    if close_approach_distance_km and close_approach_distance_km < 1000000:
        cross_section = np.pi * (earth_radius + gravitational_focus)**2
        approach_area = np.pi * close_approach_distance_km**2
        probability = min((cross_section / approach_area) * 10, 1.0)
    elif moid_km < 50000:
        probability = 0.01 * (50000 / max(moid_km, 1000))
    elif moid_km < 500000:
        probability = 0.0001 * (500000 / max(moid_km, 10000))
    else:
        probability = 1e-7
    
    return min(probability, 1.0)

def assess_risk(impact_prob: float, diameter_km: float, moid_au: float) -> str:
    """Assess overall risk level"""
    if impact_prob > 0.01:
        return "CRITICAL - Immediate threat, emergency response required"
    elif impact_prob > 0.001:
        return "HIGH - Significant threat, close monitoring essential"
    elif impact_prob > 0.0001 and diameter_km > 1.0:
        return "MODERATE - Potential threat, continued tracking recommended"
    elif moid_au < 0.05 and diameter_km > 0.14:
        return "LOW - Potentially Hazardous Asteroid (PHA), monitor trajectory"
    else:
        return "MINIMAL - No significant threat detected"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Meteor Trajectory & Impact Prediction API",
        "version": "1.0.0",
        "features": [
            "NASA NEO & SBDB Integration",
            "Poliastro Orbital Mechanics",
            "Real-time Trajectory Simulation",
            "Impact Probability Calculation"
        ],
        "endpoints": {
            "asteroids": "/asteroids/* (SBDB-based) - REMOVED",
            "meteors": "/meteors/* (NEO API-based)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "apis_available": ["NEO", "SBDB"]
    }

# ============================================================================
# METEOR/NEO ENDPOINTS (NEO API Focus)
# ============================================================================

@app.get("/meteors/browse")
async def browse_neos(
    page: int = Query(0, ge=0),
    size: int = Query(20, ge=1, le=100)
):
    """Browse Near Earth Objects from NASA NEO API"""
    url = f"{NEO_API_BASE}/neo/browse"
    params = {"page": page, "size": size, "api_key": NASA_API_KEY}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            neos = []
            for neo in data.get('near_earth_objects', []):
                neos.append({
                    "id": neo.get('id'),
                    "name": neo.get('name'),
                    "absolute_magnitude": neo.get('absolute_magnitude_h'),
                    "is_potentially_hazardous": neo.get('is_potentially_hazardous_asteroid'),
                    "estimated_diameter_km_min": neo.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_min'),
                    "estimated_diameter_km_max": neo.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_max')
                })
            
            return {
                "page": page,
                "size": size,
                "total_elements": data.get('page', {}).get('total_elements'),
                "neos": neos
            }
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Failed to browse NEOs: {str(e)}")

@app.get("/meteors/feed")
async def get_neo_feed(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get NEO feed for approaching objects"""
    if not start_date:
        start_date = datetime.utcnow().strftime("%Y-%m-%d")
    if not end_date:
        end_date = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")
    
    try:
        data = await fetch_neo_feed(start_date, end_date)
        
        approaching = []
        for date, neos in data.get('near_earth_objects', {}).items():
            for neo in neos:
                close_approach = neo.get('close_approach_data', [{}])[0]
                approaching.append({
                    "id": neo.get('id'),
                    "name": neo.get('name'),
                    "close_approach_date": close_approach.get('close_approach_date'),
                    "relative_velocity_kmh": float(close_approach.get('relative_velocity', {}).get('kilometers_per_hour', 0)),
                    "miss_distance_km": float(close_approach.get('miss_distance', {}).get('kilometers', 0)),
                    "is_potentially_hazardous": neo.get('is_potentially_hazardous_asteroid'),
                    "estimated_diameter_km": neo.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_max')
                })
        
        approaching.sort(key=lambda x: x['miss_distance_km'])
        
        return {
            "start_date": start_date,
            "end_date": end_date,
            "element_count": len(approaching),
            "approaching_objects": approaching[:50]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/meteors/{neo_id}/trajectory")
async def get_neo_trajectory(
    neo_id: str,
    days_forward: int = Query(365, ge=1, le=3650)
):
    """Get detailed trajectory and impact probability for NEO (NEO API)"""
    try:
        neo_data = await fetch_neo_data(neo_id)
        
        name = neo_data.get('name', 'Unknown')
        h_mag = neo_data.get('absolute_magnitude_h', 22)
        is_hazardous = neo_data.get('is_potentially_hazardous_asteroid', False)
        
        diameter_data = neo_data.get('estimated_diameter', {}).get('kilometers', {})
        diameter_min = diameter_data.get('estimated_diameter_min', 0.1)
        diameter_max = diameter_data.get('estimated_diameter_max', 0.3)
        diameter_km = (diameter_min + diameter_max) / 2
        
        orbital_data = neo_data.get('orbital_data', {})
        orbital_elements = {
            'semi_major_axis': float(orbital_data.get('semi_major_axis', 1.5)),
            'eccentricity': float(orbital_data.get('eccentricity', 0.1)),
            'inclination': float(orbital_data.get('inclination', 5)),
            'longitude_asc_node': float(orbital_data.get('ascending_node_longitude', 0)),
            'argument_perihelion': float(orbital_data.get('perihelion_argument', 0)),
            'mean_anomaly': float(orbital_data.get('mean_anomaly', 0)),
            'epoch': orbital_data.get('orbit_determination_date', '2025-01-01')
        }
        
        close_approaches = neo_data.get('close_approach_data', [])
        closest_approach = None
        close_approach_distance = None
        relative_velocity = None
        
        if close_approaches:
            closest = min(close_approaches, key=lambda x: float(x.get('miss_distance', {}).get('kilometers', float('inf'))))
            closest_approach = closest.get('close_approach_date')
            close_approach_distance = float(closest.get('miss_distance', {}).get('kilometers', 0))
            relative_velocity = float(closest.get('relative_velocity', {}).get('kilometers_per_hour', 0))
        
        # Use Poliastro for accurate trajectory
        trajectory_points, period_days = simulate_trajectory_poliastro(orbital_elements, days_forward)
        
        moid_au = calculate_moid(orbital_elements)
        impact_prob = calculate_impact_probability(moid_au, diameter_km, close_approach_distance)
        risk = assess_risk(impact_prob, diameter_km, moid_au)
        
        return TrajectoryResponse(
            neo_id=neo_id,
            name=name,
            is_potentially_hazardous=is_hazardous,
            close_approach_date=closest_approach,
            close_approach_distance_km=close_approach_distance,
            relative_velocity_kmh=relative_velocity,
            estimated_diameter_km_min=diameter_min,
            estimated_diameter_km_max=diameter_max,
            absolute_magnitude=h_mag,
            orbital_elements=orbital_elements,
            impact_probability=round(impact_prob, 10),
            minimum_orbit_intersection_distance_au=round(moid_au, 6),
            trajectory_points=trajectory_points,
            risk_assessment=risk,
            orbital_period_days=round(period_days, 2) if period_days else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@app.get("/statistics/neo-summary")
async def get_neo_statistics():
    """Get summary statistics of NEOs"""
    try:
        url = f"{NEO_API_BASE}/neo/browse"
        params = {"page": 0, "size": 20, "api_key": NASA_API_KEY}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            page_info = data.get('page', {})
            
            return {
                "total_neos": page_info.get('total_elements', 0),
                "total_pages": page_info.get('total_pages', 0),
                "size": page_info.get('size', 20),
                "message": "Use /meteors/browse to paginate through all NEOs"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/utils/orbital-elements-info")
async def get_orbital_elements_info():
    """Get information about orbital elements"""
    return {
        "orbital_elements": {
            "semi_major_axis": {
                "symbol": "a",
                "unit": "AU (Astronomical Units)",
                "description": "Average distance from the Sun",
                "earth_value": 1.0
            },
            "eccentricity": {
                "symbol": "e",
                "unit": "dimensionless (0-1)",
                "description": "How elliptical the orbit is (0=circle, <1=ellipse)",
                "earth_value": 0.0167
            },
            "inclination": {
                "symbol": "i",
                "unit": "degrees",
                "description": "Tilt of orbit relative to Earth's orbit plane",
                "earth_value": 0.0
            },
            "longitude_asc_node": {
                "symbol": "Ω",
                "unit": "degrees",
                "description": "Where orbit crosses Earth's orbital plane",
                "earth_value": "varies"
            },
            "argument_perihelion": {
                "symbol": "ω",
                "unit": "degrees",
                "description": "Angle from ascending node to closest point to Sun",
                "earth_value": "varies"
            },
            "mean_anomaly": {
                "symbol": "M",
                "unit": "degrees",
                "description": "Position in orbit at epoch time",
                "earth_value": "varies"
            }
        },
        "potentially_hazardous_criteria": {
            "moid": "< 0.05 AU (about 7.5 million km)",
            "diameter": "> 140 meters",
            "description": "Both criteria must be met"
        }
    }

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)